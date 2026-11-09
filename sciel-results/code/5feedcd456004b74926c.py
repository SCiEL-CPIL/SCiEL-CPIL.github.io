"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combined approach: penalty-based local refinement (L-BFGS-B) with binary search
on the outer side length, driven by a hexagonal-lattice / ring initialisation
and a simulated-annealing global search that perturbs both positions and angles.
"""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B (arrays of 2D verts). 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer (edge-wise)."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _total_penalty(params, S, center, outer_angle):
    """Total penalty: sum of squares of containment + overlap. 0 means valid.
    params: flat array [x0,y0,a0, x1,y1,a1, ...] with 3*N entries."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        if pen > 0:
            total += pen * pen
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += d * d
    return total

def _check_validity(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid (no overlaps, all contained)."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _hex_lattice_init(scale=1.0):
    """Generate a hexagonal lattice initial configuration, centered at origin."""
    u = np.array([1.5 * scale, np.sqrt(3.0)/2.0 * scale])
    v = np.array([-1.5 * scale, np.sqrt(3.0)/2.0 * scale])
    
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    idx = 0
    for i in range(4):
        for j in range(3):
            pos = i * u + j * v
            centers[idx] = pos
            angles[idx] = 0.0
            idx += 1
    
    cx_mean = np.mean(centers[:, 0])
    cy_mean = np.mean(centers[:, 1])
    centers -= np.array([cx_mean, cy_mean])
    
    return centers, angles

def _ring_init(radius=2.0):
    """Ring of 6 around center + 6 in inner ring."""
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.4)*np.cos(a), (radius*1.4)*np.sin(a)])
    arr = np.array(pts)
    arr -= arr.mean(axis=0)
    angles = np.zeros(N)
    return arr, angles

def _random_init():
    centers = np.array([[random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5)] for _ in range(N)])
    angles = np.array([random.uniform(0, 360) for _ in range(N)])
    return centers, angles

def _params_from_centers_angles(centers, angles):
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i, 0]
        params[3*i+1] = centers[i, 1]
        params[3*i+2] = angles[i]
    return params

def _refine_local(params, S, center, outer_angle, deadline):
    """L-BFGS-B local refinement minimizing penalty for fixed S."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 5000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _binary_search_min_S(init_params, center, outer_angle, deadline,
                          S_lo=3.0, S_hi=6.5, max_iters=10):
    """Binary search for minimum S such that a valid packing exists."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params.copy()
    p = _refine_local(p, S_hi, center, outer_angle, deadline)
    if _check_validity(p, S_hi, center, outer_angle):
        best_params = p.copy()
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params.copy()
        p = _refine_local(p, S_mid, center, outer_angle, deadline)
        if _check_validity(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p.copy()
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def _sa_global_search(centers_init, angles_init, S_target, center, outer_angle, deadline, steps=3000):
    """Simulated annealing to find a valid configuration at S_target,
    then return the best (lowest penalty) configuration found."""
    params = _params_from_centers_angles(centers_init, angles_init)
    cur_penalty = _total_penalty(params, S_target, center, outer_angle)
    best_params = params.copy()
    best_penalty = cur_penalty
    
    T = 0.5
    T_min = 0.001
    cooling = 0.998
    
    for step in range(steps):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        idx = random.randint(0, N-1)
        base = 3 * idx
        # Perturb position and/or angle
        if random.random() < 0.7:
            params[base] += random.gauss(0, 0.15)
            params[base+1] += random.gauss(0, 0.15)
        else:
            params[base+2] += random.gauss(0, 10)
        
        new_penalty = _total_penalty(params, S_target, center, outer_angle)
        delta = new_penalty - cur_penalty
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur_penalty = new_penalty
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_params = params.copy()
        else:
            if random.random() < 0.7:
                params[base] -= random.gauss(0, 0.15)
                params[base+1] -= random.gauss(0, 0.15)
            else:
                params[base+2] -= random.gauss(0, 10)
        
        T *= cooling
    
    return best_params, best_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_S = float('inf')
    best_params = None
    
    # Multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init(1.0))
    inits.append(_hex_lattice_init(1.05))
    inits.append(_hex_lattice_init(1.1))
    inits.append(_ring_init(2.0))
    inits.append(_random_init())
    
    for centers, angles in inits:
        if time.time() > deadline - 5:
            break
        
        init_params = _params_from_centers_angles(centers, angles)
        
        # Binary search for minimum S from this init
        S_val, p_val = _binary_search_min_S(init_params, center, outer_angle, deadline,
                                             S_lo=3.0, S_hi=6.5, max_iters=8)
        
        if S_val is not None and S_val < best_S:
            best_S = S_val
            best_params = p_val
        
        # If we found a valid config, try to improve with SA at a tighter S
        if S_val is not None and time.time() < deadline - 10:
            # Try SA at slightly smaller S to see if we can find a valid config
            S_try = S_val * 0.97
            if S_try > 3.0:
                sa_params, sa_pen = _sa_global_search(centers, angles, S_try, center, outer_angle,
                                                       deadline, steps=2000)
                if sa_pen < 1e-8:
                    # Valid at tighter S
                    S_val2, p_val2 = _binary_search_min_S(sa_params, center, outer_angle, deadline,
                                                           S_lo=S_try*0.95, S_hi=S_try, max_iters=5)
                    if S_val2 is not None and S_val2 < best_S:
                        best_S = S_val2
                        best_params = p_val2
    
    # Final local refinement at best_S
    if best_params is not None and time.time() < deadline - 5:
        refined = _refine_local(best_params, best_S, center, outer_angle, deadline)
        if _check_validity(refined, best_S, center, outer_angle):
            best_params = refined
    
    # If no valid configuration found, use a guaranteed valid one
    if best_params is None:
        centers, angles = _hex_lattice_init(1.15)
        init_params = _params_from_centers_angles(centers, angles)
        S_val, p_val = _binary_search_min_S(init_params, center, outer_angle, deadline,
                                             S_lo=3.0, S_hi=7.0, max_iters=5)
        if S_val is not None:
            best_S = S_val
            best_params = p_val
        else:
            # Fallback: very large hexagon
            best_S = 7.0
            best_params = init_params
    
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0
