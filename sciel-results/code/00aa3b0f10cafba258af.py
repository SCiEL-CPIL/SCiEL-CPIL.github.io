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

def _hex_verts_fast(cx, cy, side, angle_deg):
    """Vectorized hexagon vertices."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

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

def _smooth_max(vals, beta=30.0):
    """Smooth max using log-sum-exp."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return 0.0
    m = vals.max()
    if m < -1000:
        return 0.0
    return m + np.log(np.sum(np.exp((vals - m) * beta)) / len(vals)) / beta

def _penalty(params, S, center, outer_angle, beta=30.0):
    """Total penalty: smooth max of containment + overlap violations."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[2*i], params[2*i+1], 1.0, 0.0))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    
    # Collect all violation values
    violations = []
    for iv in inner_list:
        pen = _containment_pen(iv, outer)
        violations.append(pen)
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap(inner_list[i], inner_list[j])
            violations.append(d)
    
    # Use smooth max instead of sum of squares for smoother landscape
    max_viol = _smooth_max(np.array(violations), beta)
    # Also add a small sum of squares for global pressure
    sum_sq = sum(v * v for v in violations)
    return max_viol * max_viol + 0.01 * sum_sq

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
    pts = []
    pts.append([0, 0])
    for i in range(6):
        a = np.radians(60 * i)
        pts.append([2 * scale * np.cos(a), 2 * scale * np.sin(a)])
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

def _two_ring_init(r1=1.8, r2=3.2):
    """Two concentric rings of 6 each."""
    pts = []
    for i in range(6):
        a = np.radians(60 * i)
        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
    for i in range(6):
        a = np.radians(60 * i + 30)
        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _refine(params, S, center, outer_angle, deadline, max_iter=1000, beta=30.0):
    """L-BFGS-B local refinement with smooth penalty."""
    def obj(p):
        return _penalty(p, S, center, outer_angle, beta)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'maxfun': 20000,
                                'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _refine_progressive(params, S, center, outer_angle, deadline, max_iter=1000):
    """Progressive refinement: start with smooth penalty, tighten."""
    p = params.copy()
    # Phase 1: smooth
    if time.time() < deadline - 5:
        p = _refine(p, S, center, outer_angle, deadline, max_iter=max_iter, beta=20.0)
    # Phase 2: less smooth
    if time.time() < deadline - 3:
        p = _refine(p, S, center, outer_angle, deadline, max_iter=max_iter, beta=50.0)
    # Phase 3: sharp
    if time.time() < deadline - 1:
        p = _refine(p, S, center, outer_angle, deadline, max_iter=max_iter, beta=100.0)
    return p

def _binary_search_S(init_params, center, outer_angle, deadline,
                     S_lo=2.5, S_hi=5.0, max_iters=10):
    """Binary search for minimum S."""
    best_S = S_hi
    best_params = None
    
    # Check if S_hi works
    p = init_params.copy()
    p = _refine_progressive(p, S_hi, center, outer_angle, deadline)
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
        p = _refine_progressive(p, S_mid, center, outer_angle, deadline)
        if _check_valid(p, S_mid, center, outer_angle):
            best_S = S_mid
            best_params = p
            S_hi = S_mid
        else:
            S_lo = S_mid
    
    return best_S, best_params

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0

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
    inits.append(_hex_lattice_init(0.8))
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_ring_init(3.0))
    inits.append(_two_ring_init(1.8, 3.2))
    inits.append(_two_ring_init(2.0, 3.5))
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    
    rng = np.random.default_rng(42)
    for _ in range(12):
        r = rng.uniform(-2.5, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 25:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 20:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 10:
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
                    polished = _refine(params_found, S_found, ctr, oa, DEADLINE, max_iter=3000, beta=100.0)
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
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break

    # Final polish pass with tighter tolerance
    if best_params is not None and time.time() < DEADLINE - 5:
        final = _refine(best_params, best_S, best_ctr, best_oa, DEADLINE, max_iter=5000, beta=100.0)
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
