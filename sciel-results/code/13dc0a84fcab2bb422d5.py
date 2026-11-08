"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap(A, B):
    """SAT overlap depth between convex polygons. 0 if disjoint."""
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

def _containment_pen(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer."""
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

def _penalty(params, S, center, outer_angle):
    """Total penalty: sum of squares of containment + overlap violations."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_pen(iv, outer)
        if pen > 0:
            total += pen * pen
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap(inner_list[i], inner_list[j])
            if d > 0:
                total += d * d
    return total

def _check_valid(params, S, center, outer_angle, tol=1e-7):
    """Check if packing is valid."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_pen(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _hex_lattice_init(scale=1.0):
    """Generate hexagonal lattice positions for 12 hexagons."""
    # Honeycomb arrangement: center + 6 around + 5 more
    pts = []
    # Center
    pts.append([0, 0])
    # First ring: 6 hexagons at distance 2
    for i in range(6):
        a = np.radians(60 * i)
        pts.append([2 * scale * np.cos(a), 2 * scale * np.sin(a)])
    # Second ring: 5 more at distance 4, skipping one to keep 12 total
    for i in range(5):
        a = np.radians(60 * i + 30)
        pts.append([4 * scale * np.cos(a), 4 * scale * np.sin(a)])
    
    arr = np.array(pts[:N])
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60 * i)
        pts.append([radius * np.cos(a), radius * np.sin(a)])
    for i in range(6):
        a = np.radians(60 * i + 30)
        pts.append([radius * 1.5 * np.cos(a), radius * 1.5 * np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    xs = np.linspace(-S * 0.6, S * 0.6, N)
    return np.column_stack([xs, np.zeros(N)])

def _refine(params, S, center, outer_angle, deadline, max_iter=1000):
    """L-BFGS-B local refinement."""
    def obj(p):
        return _penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'maxfun': 20000,
                                'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _binary_search_S(init_params, center, outer_angle, deadline,
                     S_lo=2.5, S_hi=5.0, max_iters=10):
    """Binary search for minimum S."""
    best_S = S_hi
    best_params = None
    
    # Check if S_hi works
    p = init_params.copy()
    p = _refine(p, S_hi, center, outer_angle, deadline)
    if _check_valid(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params.copy()
        p = _refine(p, S_mid, center, outer_angle, deadline)
        if _check_valid(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    best_S = float('inf')
    best_params = None
    best_ctr = (0.0, 0.0)
    best_oa = 0.0

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    # Generate multiple initial configurations
    inits = []
    inits.append(_hex_lattice_init(1.0))
    inits.append(_hex_lattice_init(0.9))
    inits.append(_hex_lattice_init(1.1))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_ring_init(3.0))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    
    rng = np.random.default_rng(42)
    for _ in range(8):
        r = rng.uniform(-2, 2, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 20:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 15:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 8:
                    break
                
                init_params = np.zeros(2 * N)
                for i in range(N):
                    init_params[2*i] = init_pts[i, 0]
                    init_params[2*i+1] = init_pts[i, 1]
                
                S_found, params_found = _binary_search_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=2.5, S_hi=5.0, max_iters=10
                )
                
                if params_found is not None and S_found < best_S:
                    # Final high-precision polish
                    polished = _refine(params_found, S_found, ctr, oa, DEADLINE, max_iter=2000)
                    if _check_valid(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_params = polished
                        best_ctr = ctr
                        best_oa = oa
                    else:
                        best_S = S_found
                        best_params = params_found
                        best_ctr = ctr
                        best_oa = oa
                
                # Early exit if excellent solution found
                if best_S < 3.2:
                    break
            if time.time() > DEADLINE - 8:
                break
        if time.time() > DEADLINE - 8:
            break

    # Final polish pass with tighter tolerance
    if best_params is not None and time.time() < DEADLINE - 5:
        final = _refine(best_params, best_S, best_ctr, best_oa, DEADLINE, max_iter=3000)
        if _check_valid(final, best_S, best_ctr, best_oa):
            best_params = final

    if best_params is None:
        # Fallback
        grid = _hex_lattice_init(1.0)
        inner = [[float(grid[i, 0]), float(grid[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    inner = [[float(best_params[2*i]), float(best_params[2*i+1]), 0.0] for i in range(N)]
    return inner, (float(best_ctr[0]), float(best_ctr[1])), float(best_S), float(best_oa)

# EVOLVE-BLOCK-END
