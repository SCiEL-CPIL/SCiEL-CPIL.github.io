"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _smooth_penalty(params, S, center, outer_angle, beta=50.0):
    """Smooth surrogate using log-sum-exp for better gradient info."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    violations = []
    for iv in inner_list:
        pen = _containment_violation(iv, outer)
        violations.append(pen)
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            violations.append(d)
    
    if not violations:
        return 0.0
    
    v = np.array(violations)
    max_v = v.max()
    if max_v < 1e-10:
        return 0.0
    # Log-sum-exp for smooth max
    shifted = v - max_v
    lse = max_v + np.log(np.sum(np.exp(beta * shifted))) / beta
    return lse

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

def _refine_local(params, S, center, outer_angle, deadline, max_iter=500):
    """L-BFGS-B local refinement with strict penalty."""
    def obj(p):
        return _total_penalty(p, S, center, outer_angle)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'maxfun': max_iter * 20,
                                'ftol': 1e-16, 'gtol': 1e-12})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _refine_local_smooth(params, S, center, outer_angle, deadline, max_iter=300):
    """L-BFGS-B with smooth surrogate for better convergence."""
    def obj(p):
        return _smooth_penalty(p, S, center, outer_angle, beta=50.0)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'maxfun': max_iter * 20,
                                'ftol': 1e-14, 'gtol': 1e-10})
        if time.time() > deadline:
            return params
        # Follow up with sharp penalty
        return _refine_local(res.x, S, center, outer_angle, deadline, max_iter=200)
    except Exception:
        return params

def _de_global_search(S, center, outer_angle, deadline, seed=42):
    """Differential evolution in 36D space (positions + angles)."""
    t_start = time.time()
    remaining = deadline - t_start
    if remaining < 5:
        return None
    
    # Bounds: positions in [-S, S], angles in [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-S, S))  # x
        bounds.append((-S, S))  # y
        bounds.append((0, 360))  # angle
    
    # Use smooth penalty for DE (better landscape)
    def de_obj(params):
        return _smooth_penalty(params, S, center, outer_angle, beta=30.0)
    
    # Time-limited DE
    max_iter = max(10, int(remaining * 2))  # ~2 iterations per second
    popsize = min(15, max(5, int(remaining * 0.3)))
    
    try:
        res = differential_evolution(
            de_obj, bounds,
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-6,
            seed=seed,
            init='sobol',
            polish=False,  # We'll polish separately
            updating='deferred'
        )
        if time.time() > deadline - 5:
            return None
        return res.x
    except Exception:
        return None

def _binary_search_min_S(init_params, center, outer_angle, deadline,
                          S_lo=3.0, S_hi=7.0, max_iters=6):
    """Binary search for minimum S such that a valid packing exists."""
    best_S = S_hi
    best_params = None
    
    # First check if S_hi works
    p = init_params.copy()
    p = _refine_local(p, S_hi, center, outer_angle, deadline)
    if _check_validity(p, S_hi, center, outer_angle):
        best_params = p
        best_S = S_hi
    else:
        return None, None
    
    for _ in range(max_iters):
        if time.time() > deadline - 8:
            break
        S_mid = (S_lo + S_hi) / 2.0
        p = best_params.copy() if best_params is not None else init_params.copy()
        p = _refine_local(p, S_mid, center, outer_angle, deadline)
        if _check_validity(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin before 150s limit

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    # Generate initial configurations
    inits = []
    
    # Grid-like initializations
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            pos = []
            for r in range(nr):
                for c in range(nc):
                    x = c * 2.0 + (1.0 if r % 2 else 0.0)
                    y = r * np.sqrt(3.0)
                    pos.append([x, y])
            arr = np.array(pos)
            arr -= arr.mean(axis=0)
            # Convert to full params with angle 0
            full = np.zeros(3 * N)
            for i in range(N):
                full[3*i] = arr[i, 0]
                full[3*i+1] = arr[i, 1]
                full[3*i+2] = 0.0
            inits.append(full)
    
    # Ring initialization
    for radius in [2.0, 2.5, 3.0]:
        pts = []
        for i in range(6):
            a = np.radians(60*i)
            pts.append([radius*np.cos(a), radius*np.sin(a)])
        for i in range(6):
            a = np.radians(60*i + 30)
            pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
        arr = np.array(pts)
        arr -= arr.mean(axis=0)
        full = np.zeros(3 * N)
        for i in range(N):
            full[3*i] = arr[i, 0]
            full[3*i+1] = arr[i, 1]
            full[3*i+2] = 0.0
        inits.append(full)
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(8):
        r = rng.uniform(-2.5, 2.5, size=(N, 2))
        angles = rng.uniform(0, 360, size=N)
        full = np.zeros(3 * N)
        for i in range(N):
            full[3*i] = r[i, 0]
            full[3*i+1] = r[i, 1]
            full[3*i+2] = angles[i]
        inits.append(full)
    
    # Strategy 1: DE global search followed by binary search + local refinement
    for oa in outer_angles:
        if time.time() > DEADLINE - 30:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 25:
                break
            
            # Run DE to find a good starting point
            de_result = _de_global_search(5.0, ctr, oa, DEADLINE - 10, seed=int(oa*10)+42)
            
            if de_result is not None:
                # Use DE result as initial point for binary search
                S_found, params_found = _binary_search_min_S(
                    de_result, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=6.5, max_iters=5
                )
                if params_found is not None and S_found < best_S:
                    # Final high-precision polish
                    polished = _refine_local(params_found, S_found, ctr, oa, DEADLINE, max_iter=800)
                    if _check_validity(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
            
            # Strategy 2: Try each initialization with binary search
            for init in inits:
                if time.time() > DEADLINE - 15:
                    break
                
                S_found, params_found = _binary_search_min_S(
                    init.copy(), ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=6.0, max_iters=4
                )
                
                if params_found is not None and S_found < best_S:
                    # Final high-precision polish
                    polished = _refine_local(params_found, S_found, ctr, oa, DEADLINE, max_iter=600)
                    if _check_validity(polished, S_found, ctr, oa):
                        best_S = S_found
                        best_result = (S_found, polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
                
                # Early exit if very good
                if best_S < 3.5:
                    break
            
            if time.time() > DEADLINE - 15:
                break
        
        if time.time() > DEADLINE - 15:
            break
    
    # Strategy 3: If we have time, try smooth refinement on best result
    if best_result is not None and time.time() < DEADLINE - 10:
        S, params, ctr, oa = best_result
        smoothed = _refine_local_smooth(params.copy(), S, ctr, oa, DEADLINE, max_iter=400)
        if _check_validity(smoothed, S, ctr, oa):
            # Try to reduce S slightly
            S_try = S * 0.999
            p_try = _refine_local(smoothed, S_try, ctr, oa, DEADLINE, max_iter=300)
            if _check_validity(p_try, S_try, ctr, oa):
                best_S = S_try
                best_result = (S_try, p_try, ctr, oa)
            else:
                best_result = (S, smoothed, ctr, oa)
    
    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        pos = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 else 0.0)
                y = r * np.sqrt(3.0)
                pos.append([x, y])
        arr = np.array(pos)
        arr -= arr.mean(axis=0)
        inner = [[float(arr[i,0]), float(arr[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.5, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
