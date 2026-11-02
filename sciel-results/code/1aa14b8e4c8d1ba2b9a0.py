"""
Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points.

Strategy:
1. Exploit symmetry by parameterizing points as nested regular polygons.
2. Optimize the low-dimensional parameters (radii, angles) using Differential Evolution.
3. Refine the best symmetric solutions using full-dimensional L-BFGS-B to break symmetry if beneficial.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
DEADLINE = 115.0  # Safety margin for 120s limit

# Pre-compute all triples for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)
_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Center and scale to unit area for stability? 
    # The task says "normalized by the point set convex hull area".
    # So we just need min_area / hull_area.
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    xi = x[_triple_i]
    xj = x[_triple_j]
    xk = x[_triple_k]
    yi = y[_triple_i]
    yj = y[_triple_j]
    yk = y[_triple_k]

    cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    areas = 0.5 * cross
    min_area = np.min(areas)

    if min_area < 1e-15:
        return 0.0

    return min_area / hull_area


def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _generate_symmetric_config(params):
    """
    Generate points from symmetric parameters.
    Supports various ring structures.
    params format depends on the specific generator called, but we'll use a unified approach
    where we try a few specific topologies.
    
    Actually, it's better to have separate objective functions for different topologies
    to keep bounds simple.
    """
    pass

def _make_ring(n, radius, angle_offset=0.0):
    """Generate n points on a regular polygon ring."""
    if n == 0:
        return np.array([]).reshape(0, 2)
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _config_1_center_12_ring(params):
    """1 center + 12 ring. params: [r_ring, angle_offset]"""
    r = params[0]
    ang = params[1]
    center = np.array([[0.0, 0.0]])
    ring = _make_ring(12, r, ang)
    return np.vstack([center, ring])

def _config_2_rings(params):
    """Two rings. params: [r1, n1, r2, n2, ang1, ang2] is hard for DE if n is int.
    We fix the counts. 
    Let's try 6 outer + 7 inner.
    params: [r_outer, r_inner, ang_outer, ang_inner]
    """
    r_out = params[0]
    r_in = params[1]
    ang_out = params[2]
    ang_in = params[3]
    outer = _make_ring(6, r_out, ang_out)
    inner = _make_ring(7, r_in, ang_in)
    return np.vstack([outer, inner])

def _config_3_rings(params):
    """1 center + 5 mid + 7 outer.
    params: [r_mid, r_out, ang_mid, ang_out]
    """
    r_mid = params[0]
    r_out = params[1]
    ang_mid = params[2]
    ang_out = params[3]
    center = np.array([[0.0, 0.0]])
    mid = _make_ring(5, r_mid, ang_mid)
    outer = _make_ring(7, r_out, ang_out)
    return np.vstack([center, mid, outer])

def _config_single_ring(params):
    """13-gon. params: [r, ang]"""
    r = params[0]
    ang = params[1]
    return _make_ring(13, r, ang)

def _config_3_3_7(params):
    """3 inner + 3 mid + 7 outer.
    params: [r_in, r_mid, r_out, ang_in, ang_mid, ang_out]
    """
    r_in, r_mid, r_out = params[0], params[1], params[2]
    ang_in, ang_mid, ang_out = params[3], params[4], params[5]
    inner = _make_ring(3, r_in, ang_in)
    mid = _make_ring(3, r_mid, ang_mid)
    outer = _make_ring(7, r_out, ang_out)
    return np.vstack([inner, mid, outer])


def construct():
    t_start = time.time()
    best_score = -float('inf')
    best_params = None
    rng = np.random.RandomState(42)

    def _update(pts):
        nonlocal best_score, best_params
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()

    def _time_ok(margin=10):
        return (time.time() - t_start) < (DEADLINE - margin)

    # List of (generator_func, bounds, initial_guess)
    # Bounds are for the parameters of the symmetric config
    # We want to explore a reasonable range of radii. 
    # Since we normalize by hull area, the absolute scale doesn't matter for the score,
    # but it matters for the optimization landscape. We can fix the outermost radius 
    # or let it vary. Let's let it vary but keep it in a sane range.
    
    search_spaces = []
    
    # 1. 1 center + 12 ring
    # r in [0.5, 2.0], ang in [0, pi/6] (symmetry of 12-gon is pi/6)
    search_spaces.append((_config_1_center_12_ring, [(0.5, 2.0), (0, np.pi/6)], [1.0, 0.0]))
    
    # 2. 6 outer + 7 inner
    # r_out in [0.8, 2.0], r_in in [0.1, r_out], angles
    # To handle dependent bounds (r_in < r_out), we can just use broad bounds and rely on the score function 
    # to penalize overlaps or just let DE explore. 
    # Actually, if r_in > r_out, the "inner" is outside. That's fine, it's just two rings.
    search_spaces.append((_config_2_rings, [(0.5, 2.0), (0.1, 2.0), (0, np.pi/3), (0, np.pi/7)], [1.5, 0.8, 0.0, 0.0]))
    
    # 3. 1 center + 5 mid + 7 outer
    search_spaces.append((_config_3_rings, [(0.2, 1.5), (0.8, 2.0), (0, np.pi/5), (0, np.pi/7)], [0.7, 1.5, 0.0, 0.0]))
    
    # 4. Single 13-gon
    search_spaces.append((_config_single_ring, [(0.5, 2.0), (0, np.pi/13)], [1.0, 0.0]))
    
    # 5. 3 inner + 3 mid + 7 outer
    search_spaces.append((_config_3_3_7, [(0.1, 1.0), (0.4, 1.5), (0.8, 2.0), (0, np.pi/3), (0, np.pi/3), (0, np.pi/7)], [0.5, 1.0, 1.5, 0.0, 0.0, 0.0]))
    
    # 6. 2 center-ish + 11 ring? No, 1 center is better.
    # 7. 4 inner + 9 outer
    def _config_4_9(params):
        r_in, r_out = params[0], params[1]
        ang_in, ang_out = params[2], params[3]
        inner = _make_ring(4, r_in, ang_in)
        outer = _make_ring(9, r_out, ang_out)
        return np.vstack([inner, outer])
    search_spaces.append((_config_4_9, [(0.2, 1.5), (0.8, 2.0), (0, np.pi/4), (0, np.pi/9)], [0.8, 1.5, 0.0, 0.0]))
    
    # 8. 5 inner + 8 outer
    def _config_5_8(params):
        r_in, r_out = params[0], params[1]
        ang_in, ang_out = params[2], params[3]
        inner = _make_ring(5, r_in, ang_in)
        outer = _make_ring(8, r_out, ang_out)
        return np.vstack([inner, outer])
    search_spaces.append((_config_5_8, [(0.2, 1.5), (0.8, 2.0), (0, np.pi/5), (0, np.pi/8)], [0.8, 1.5, 0.0, 0.0]))

    # Phase 1: Differential Evolution on symmetric parameters
    for gen_func, bounds, init_guess in search_spaces:
        if not _time_ok(20):
            break
            
        def _obj_sym(params):
            pts = gen_func(params)
            return -_score_points(pts)
        
        try:
            # Use DE to find best symmetric config
            # popsize and maxiter tuned for speed
            res = differential_evolution(
                _obj_sym,
                bounds,
                seed=42,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='deferred',
                workers=1
            )
            best_sym_params = res.x
            best_pts = gen_func(best_sym_params)
            _update(best_pts)
            
        except Exception as e:
            pass

    # Phase 2: Local Refinement (Break Symmetry)
    # Take the best symmetric points found so far and try to improve them 
    # using full-dimensional L-BFGS-B.
    if best_params is not None and _time_ok(30):
        # Create several perturbed starts
        starts = [best_params.flatten().copy()]
        for _ in range(5):
            pert = best_params + rng.normal(0, 0.05, size=(N, 2))
            starts.append(pert.flatten())
            
        for s in starts:
            if not _time_ok(10):
                break
            try:
                res = minimize(
                    _neg_score,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12}
                )
                cand = res.x.reshape(N, 2)
                _update(cand)
            except Exception:
                pass

    # Phase 3: Simulated Annealing / Random Walk on full space from best
    # If we have time, do a quick SA run to escape any remaining local minima.
    if best_params is not None and _time_ok(40):
        pts = best_params.copy()
        current_score = _score_points(pts)
        best_sa_score = current_score
        best_sa_pts = pts.copy()
        
        t = 0.1
        step = 0.05
        max_iter = 5000
        
        for i in range(max_iter):
            if not _time_ok(10):
                break
            
            new_pts = pts + rng.normal(0, step, size=pts.shape)
            new_score = _score_points(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_sa_score:
                    best_sa_score = current_score
                    best_sa_pts = pts.copy()
            else:
                delta = current_score - new_score
                if delta < 0.05: # Only accept small decreases
                    prob = np.exp(-delta / t)
                    if rng.random() < prob:
                        pts = new_pts
                        current_score = new_score
            
            t = max(0.01, t * 0.999)
            step = max(0.01, step * 0.999)
        
        _update(best_sa_pts)
        
        # Final polish on the SA result
        if _time_ok(10):
            try:
                res = minimize(
                    _neg_score,
                    best_sa_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12}
                )
                cand = res.x.reshape(N, 2)
                _update(cand)
            except Exception:
                pass

    if best_params is None:
        # Fallback: Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x0 = np.column_stack([np.cos(angles), np.sin(angles)])
        return x0
        
    return best_params
