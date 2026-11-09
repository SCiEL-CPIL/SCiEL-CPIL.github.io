"""Unit regular hexagons inside a regular hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_HEX_COS = np.array([np.cos(np.radians(60 * i)) for i in range(6)])
_HEX_SIN = np.array([np.sin(np.radians(60 * i)) for i in range(6)])

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Rotate the base hexagon vertices
    rx = _HEX_COS * ca - _HEX_SIN * sa
    ry = _HEX_COS * sa + _HEX_SIN * ca
    return np.column_stack([cx + side * rx, cy + side * ry])

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

def _precompute_outer_normals(outer_verts):
    """Precompute edge normals and origins for containment checks."""
    n = len(outer_verts)
    origins = outer_verts.copy()
    normals = np.zeros((n, 2))
    for i in range(n):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % n]
        edge = v2 - v1
        el = np.linalg.norm(edge)
        if el > 1e-12:
            normals[i] = np.array([-edge[1], edge[0]]) / el
    return origins, normals

def _containment_violation(inner_verts, origins, normals):
    """Sum of distances by which inner verts fall outside outer (edge-wise)."""
    pen = 0.0
    for iv in inner_verts:
        # Vectorized: compute signed distances to all edges
        diffs = iv[None, :] - origins  # shape (6, 2)
        sds = diffs @ normals.T  # shape (6,)
        neg = np.maximum(-sds, 0)
        pen += neg.sum()
    return pen

def _total_penalty(params, S, center, outer_angle, _outer_cache=None):
    """Total penalty: sum of squares of containment + overlap. 0 means valid."""
    if _outer_cache is not None:
        outer, origins, normals = _outer_cache
    else:
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        origins, normals = _precompute_outer_normals(outer)
    
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    
    total = 0.0
    for iv in inner_list:
        pen = _containment_violation(iv, origins, normals)
        if pen > 0:
            total += pen * pen
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                total += d * d
    return total

def _check_validity(params, S, center, outer_angle, tol=1e-7):
    """Check if packing is valid (no overlaps, all contained)."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    origins, normals = _precompute_outer_normals(outer)
    
    for iv in inner_list:
        if _containment_violation(iv, origins, normals) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _polygon_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

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

def _refine_local(params, S, center, outer_angle, deadline, maxiter=300, maxfun=8000, ftol=1e-14, gtol=1e-10):
    """L-BFGS-B local refinement with squared penalty."""
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    origins, normals = _precompute_outer_normals(outer)
    outer_cache = (outer, origins, normals)
    
    def obj(p):
        return _total_penalty(p, S, center, outer_angle, outer_cache)
    
    try:
        res = minimize(obj, params, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxfun,
                                'ftol': ftol, 'gtol': gtol})
        if time.time() > deadline:
            return params
        return res.x
    except Exception:
        return params

def _polish(params, S, center, outer_angle, deadline):
    """High-precision polish: tighten S slightly and re-optimize."""
    best_params = params.copy()
    best_S = S
    
    # Aggressive multi-stage polish: try many small reductions
    for stage in range(8):
        if time.time() > deadline - 2:
            break
        # Decreasing step sizes
        frac = 0.003 * (0.7 ** stage)
        S_try = best_S * (1.0 - frac)
        p = _refine_local(best_params, S_try, center, outer_angle, deadline,
                          maxiter=800, maxfun=20000, ftol=1e-16, gtol=1e-13)
        if _check_validity(p, S_try, center, outer_angle, tol=1e-8):
            best_params = p
            best_S = S_try
        else:
            # Try even smaller step
            S_try2 = best_S * (1.0 - frac * 0.5)
            p2 = _refine_local(best_params, S_try2, center, outer_angle, deadline,
                               maxiter=800, maxfun=20000, ftol=1e-16, gtol=1e-13)
            if _check_validity(p2, S_try2, center, outer_angle, tol=1e-8):
                best_params = p2
                best_S = S_try2
            else:
                break
    
    return best_S, best_params

def _binary_search_min_S(init_params, center, outer_angle, deadline,
                          S_lo=3.0, S_hi=5.5, max_iters=8):
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
        if time.time() > deadline - 5:
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
    DEADLINE = t0 + 115.0  # safety margin before 120s limit

    best_S = float('inf')
    best_result = None

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    # Generate initial configurations - prioritize the 4x3 lattice
    inits = []
    # 4x3 and 3x4 lattices first (most natural for hex packing)
    inits.append(_make_grid(4, 3))
    inits.append(_make_grid(3, 4))
    inits.append(_make_grid(2, 6))
    inits.append(_make_grid(6, 2))
    # Ring configurations
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    inits.append(_ring_init(1.8))
    inits.append(_ring_init(2.2))
    # Linear
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    
    rng = np.random.default_rng(42)
    for _ in range(6):
        r = rng.uniform(0, 3.0, size=(N, 2))
        inits.append(r - r.mean(axis=0))
    
    # Structured random with hexagonal symmetry hints
    rng2 = np.random.default_rng(123)
    for _ in range(3):
        r = rng2.uniform(0, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    for oa in outer_angles:
        if time.time() > DEADLINE - 25:
            break
        for ctr in centers:
            if time.time() > DEADLINE - 20:
                break
            for init_pts in inits:
                if time.time() > DEADLINE - 15:
                    break
                
                init_params = np.zeros(3 * N)
                for i in range(N):
                    init_params[3*i] = init_pts[i, 0]
                    init_params[3*i+1] = init_pts[i, 1]
                    init_params[3*i+2] = 0.0
                
                # Binary search for minimum S
                S_found, params_found = _binary_search_min_S(
                    init_params, ctr, oa, DEADLINE,
                    S_lo=3.0, S_hi=5.5, max_iters=8
                )
                
                if params_found is not None and S_found < best_S:
                    # Final polish to tighten S further
                    S_polished, params_polished = _polish(params_found, S_found, ctr, oa, DEADLINE)
                    if _check_validity(params_polished, S_polished, ctr, oa, tol=1e-7):
                        best_S = S_polished
                        best_result = (S_polished, params_polished, ctr, oa)
                    else:
                        best_S = S_found
                        best_result = (S_found, params_found, ctr, oa)
                
                # Early exit if we found a very good solution
                if best_S < 3.25:
                    break
            if time.time() > DEADLINE - 15:
                break
        if time.time() > DEADLINE - 15:
            break

    # Final global polish on the best result - multiple passes
    if best_result is not None and time.time() < DEADLINE - 5:
        S, params, ctr, oa = best_result
        for _ in range(3):
            if time.time() > DEADLINE - 2:
                break
            S_final, params_final = _polish(params, S, ctr, oa, DEADLINE)
            if _check_validity(params_final, S_final, ctr, oa, tol=1e-7):
                best_S = S_final
                best_result = (S_final, params_final, ctr, oa)
                S, params = S_final, params_final

    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        grid = _make_grid(4, 3)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
