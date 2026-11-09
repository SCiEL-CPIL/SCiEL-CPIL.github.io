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
    """Total penalty: sum of squares of containment + overlap. 0 means valid."""
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

def _total_penalty_xy(params, S, center, outer_angle):
    """Total penalty for fixed angle 0. params is [x0, y0, x1, y1, ...]"""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
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

def _check_validity_xy(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid (no overlaps, all contained) for fixed angle 0."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

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

def _physics_refine_xy(params, S, center, outer_angle, deadline):
    """Physics-based refinement for fixed angle 0."""
    params = params.copy()
    step_size = 0.1
    max_iters = 200
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate forces
        forces = np.zeros(2 * N)
        inner_list = []
        for i in range(N):
            inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        
        # Containment forces
        for i in range(N):
            iv = inner_list[i]
            for iv_pt in iv:
                for k in range(6):
                    v1 = outer[k]
                    v2 = outer[(k + 1) % 6]
                    edge = v2 - v1
                    el = np.linalg.norm(edge)
                    if el < 1e-12: continue
                    nrm = np.array([-edge[1], edge[0]]) / el
                    sd = np.dot(iv_pt - v1, nrm)
                    if sd < 0:
                        forces[2*i:2*i+2] += -nrm * (-sd)
        
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                d = _polygon_overlap_depth(inner_list[i], inner_list[j])
                if d > 0:
                    c_i = np.array([params[2*i], params[2*i+1]])
                    c_j = np.array([params[2*j], params[2*j+1]])
                    diff = c_i - c_j
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        diff = np.array([1.0, 0.0])
                        dist = 1.0
                    dir_vec = diff / dist
                    forces[2*i:2*i+2] += dir_vec * d
                    forces[2*j:2*j+2] -= dir_vec * d
        
        # Normalize and apply
        f_norm = np.linalg.norm(forces)
        if f_norm < 1e-8:
            break
        forces = forces / f_norm
        
        # Adaptive step size
        new_params = params + step_size * forces
        new_pen = _total_penalty_xy(new_params, S, center, outer_angle)
        old_pen = _total_penalty_xy(params, S, center, outer_angle)
        
        if new_pen < old_pen:
            params = new_params
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size *= 0.7
        
        if step_size < 1e-6:
            break
            
    return params

def _refine_local_xy(params, S, center, outer_angle, deadline):
    """Hybrid refinement: Physics then L-BFGS."""
    p_phys = _physics_refine_xy(params, S, center, outer_angle, deadline)
    
    def obj(p):
        return _total_penalty_xy(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, p_phys, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 5000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return p_phys
        if _total_penalty_xy(res.x, S, center, outer_angle) < _total_penalty_xy(p_phys, S, center, outer_angle):
            return res.x
        else:
            return p_phys
    except Exception:
        return p_phys

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def _refine_local(params, S, center, outer_angle, deadline):
    """L-BFGS-B local refinement with strict penalty."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _binary_search_min_S_xy(init_params_xy, center, outer_angle, deadline,
                             S_lo=3.0, S_hi=6.0, max_iters=8):
    """Binary search for minimum S for fixed angle 0 packing."""
    best_S = S_hi
    best_params = None
    
    p = init_params_xy.copy()
    p = _refine_local_xy(p, S_hi, center, outer_angle, deadline)
    if _check_validity_xy(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 5:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params_xy.copy()
        p = _refine_local_xy(p, S_mid, center, outer_angle, deadline)
        if _check_validity_xy(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def _high_precision_polish(params_xy, S, center, outer_angle, deadline):
    """Final high-precision polish: tighten L-BFGS tolerances and add small perturbations."""
    if time.time() > deadline - 2:
        return params_xy
    
    def obj(p):
        return _total_penalty_xy(p, S, center, outer_angle)
    
    best_params = params_xy.copy()
    best_pen = _total_penalty_xy(params_xy, S, center, outer_angle)
    
    rng = np.random.default_rng(123)
    for attempt in range(3):
        if time.time() > deadline - 1:
            break
        if attempt == 0:
            p_start = params_xy.copy()
        else:
            p_start = params_xy + rng.normal(0, 0.01, size=params_xy.shape)
        
        try:
            res = minimize(obj, p_start, method='L-BFGS-B',
                           options={'maxiter': 500, 'maxfun': 10000,
                                    'ftol': 1e-15, 'gtol': 1e-12})
            if time.time() > deadline - 1:
                break
            pen = _total_penalty_xy(res.x, S, center, outer_angle)
            if pen < best_pen:
                best_pen = pen
                best_params = res.x
        except Exception:
            pass
    
    return best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    # Added 15 and 45 degrees as per inspiration, keeping 0 and 30
    outer_angles = [0.0, 30.0, 15.0, 45.0]

    # Generate initial configurations
    inits = []
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_ring_init(3.0))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    inits.append(_linear_init(3.0))
    
    # Two-row staggered patterns
    for offset in [0.0, 0.5, 1.0]:
        pts = []
        for i in range(6):
            x = i * 2.0 + offset
            y = 0.0 if i < 3 else np.sqrt(3.0)
            pts.append([x, y])
        arr = np.array(pts)
        arr -= arr.mean(axis=0)
        inits.append(arr)
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.uniform(0, 3.0, size=(N, 2))
        inits.append(r - r.mean(axis=0))
    
    # Clustered random
    rng2 = np.random.default_rng(7)
    for _ in range(3):
        centers_r = rng2.uniform(-1.5, 1.5, size=(4, 2))
        pts = []
        for ci in range(4):
            for k in range(3):
                a = rng2.uniform(0, 2*np.pi)
                r = rng2.uniform(0, 0.8)
                pts.append([centers_r[ci, 0] + r*np.cos(a), centers_r[ci, 1] + r*np.sin(a)])
        arr = np.array(pts)
        arr -= arr.mean(axis=0)
        inits.append(arr)

    for oa in outer_angles:
        if time.time() > DEADLINE - 25:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 20:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 12:
                    break
                
                init_params_xy = np.zeros(2 * N)
                for i in range(N):
                    init_params_xy[2*i] = init_pts[i, 0]
                    init_params_xy[2*i+1] = init_pts[i, 1]
                
                # Increased max_iters to 10 for tighter binary search
                S_found_xy, params_found_xy = _binary_search_min_S_xy(
                    init_params_xy, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.5, max_iters=10
                )
                
                if params_found_xy is not None and S_found_xy < best_S:
                    polished_xy = _high_precision_polish(params_found_xy, S_found_xy, ctr, oa, DEADLINE)
                    
                    params_full = np.zeros(3 * N)
                    for i in range(N):
                        params_full[3*i] = polished_xy[2*i]
                        params_full[3*i+1] = polished_xy[2*i+1]
                        params_full[3*i+2] = 0.0
                    
                    polished = _refine_local(params_full, S_found_xy, ctr, oa, DEADLINE)
                    if _check_validity(polished, S_found_xy, ctr, oa):
                        best_S = S_found_xy
                        best_result = (S_found_xy, polished, ctr, oa)
                    else:
                        best_S = S_found_xy
                        best_result = (S_found_xy, params_full, ctr, oa)

                # Early exit threshold tightened slightly
                if best_S < 3.3:
                    break
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break

    if best_result is None:
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
