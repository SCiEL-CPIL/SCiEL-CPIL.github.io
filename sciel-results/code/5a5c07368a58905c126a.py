"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

def _all_triangle_areas(pts):
    """Compute all triangle areas for N points, vectorized."""
    n = len(pts)
    if n < 3:
        return np.array([0.0])
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas.append(0.5 * cross)
    if not areas:
        return np.array([0.0])
    return np.concatenate(areas)

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    areas = _all_triangle_areas(pts)
    return float(np.min(areas)) if len(areas) > 0 else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_area(pts, alpha):
    """Correct smooth approximation: -1/alpha * log(sum(exp(-alpha * a_i))) over all triangles."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    # Numerically stable log-sum-exp
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 0.0
    shifted = -alpha * (areas - min_a)
    log_sum = np.log(np.sum(np.exp(shifted))) + alpha * min_a
    return -log_sum / alpha

def _smooth_score(pts, alpha):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_area(pts, alpha) / hull_a

def _neg_smooth_score(params, alpha):
    pts = params.reshape(-1, 2)
    s = _smooth_score(pts, alpha)
    return -s

def _make_symmetric_points(params):
    """
    Construct 13 points: 6 outer hexagon + 6 inner hexagon + 1 center.
    params: [r_outer, r_inner, theta_outer, theta_inner, cx, cy]
    """
    r_out, r_in, th_out, th_in, cx, cy = params
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_out
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_in
    pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    pts_c = np.array([[cx, cy]])
    return np.vstack([pts_out, pts_in, pts_c])

def _neg_symmetric_score(params):
    pts = _make_symmetric_points(params)
    return -_score(pts)

def _neg_symmetric_smooth_score(params, alpha):
    pts = _make_symmetric_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_area(pts, alpha)
    return -(smooth_min / hull_a)

def _make_ring_points(params):
    """
    Construct 13 points from 3 concentric rings: 5 + 5 + 3.
    params: [r1, r2, r3, th1, th2, th3]
    """
    r1, r2, r3, th1, th2, th3 = params
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + th1
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + th2
    a3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + th3
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    return np.vstack([p1, p2, p3])

def _neg_ring_score(params):
    pts = _make_ring_points(params)
    return -_score(pts)

def _neg_ring_smooth_score(params, alpha):
    pts = _make_ring_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_area(pts, alpha)
    return -(smooth_min / hull_a)

def _local_refine(pts, time_budget, deadline):
    """Refine a point set using progressive smooth annealing + Nelder-Mead."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Progressive smooth annealing
    for alpha in [20.0, 50.0, 150.0, 400.0, 1000.0]:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                lambda p, a=alpha: _neg_smooth_score(p, a),
                current_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            new_pts = res.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Diverse initializations ===
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 12-gon + center
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # Hexagonal symmetric (6+6+1)
    for r_in in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for th_in in [0.0, np.pi/6.0, np.pi/12.0]:
            params = [1.0, r_in, 0.0, th_in, 0.0, 0.0]
            inits.append(_make_symmetric_points(params))
    
    # Ring structure (5+5+3)
    for r2 in [0.5, 0.65, 0.8]:
        for r3 in [0.2, 0.4, 0.6]:
            for th3 in [0.0, np.pi/6.0]:
                params = [1.0, r2, r3, 0.0, np.pi/5.0, th3]
                inits.append(_make_ring_points(params))
    
    # Jittered
    for jitter in [0.02, 0.08, 0.15]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    
    # Random in unit disk
    for _ in range(6):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Evaluate initializations
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 100.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # === PHASE 2: Differential Evolution for global search ===
    if time.time() < deadline - 60.0:
        bounds = [(-1.5, 1.5)] * (2 * N)
        try:
            de_result = differential_evolution(
                _neg_score,
                bounds=bounds,
                seed=42,
                maxiter=80,
                popsize=12,
                tol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=-1
            )
            de_pts = de_result.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # === PHASE 3: Local refinement of top candidates ===
    # Collect top candidates
    candidates = []
    for s, p in scored_inits[:5]:
        candidates.append(p)
    candidates.append(best_pts)
    
    # Add DE result if available
    if time.time() < deadline - 40.0:
        for c in candidates:
            if time.time() > deadline - 15.0:
                break
            refined_pts, refined_score = _local_refine(c, time.time() - start_time, deadline)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
    
    # === PHASE 4: Symmetric space optimization ===
    if time.time() < deadline - 30.0:
        # Hexagonal symmetric (6+6+1)
        sym_inits = [
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.65, 0.0, np.pi/6.0, 0.0, 0.0],
            [1.0, 0.75, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.55, 0.0, np.pi/12.0, 0.0, 0.0],
            [1.0, 0.4, 0.0, np.pi/6.0, 0.0, 0.0],
        ]
        for sym_init in sym_inits:
            if time.time() > deadline - 10.0:
                break
            current_params = np.array(sym_init)
            current_score = _score(_make_symmetric_points(current_params))
            
            for alpha in [30.0, 100.0, 300.0]:
                if time.time() > deadline - 8.0:
                    break
                try:
                    res = minimize(
                        lambda p, a=alpha: _neg_symmetric_smooth_score(p, a),
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    new_params = res.x
                    new_score = _score(_make_symmetric_points(new_params))
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                except Exception:
                    pass
            
            if time.time() < deadline - 5.0:
                try:
                    res = minimize(
                        _neg_symmetric_score,
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13}
                    )
                    new_params = res.x
                    new_score = _score(_make_symmetric_points(new_params))
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                except Exception:
                    pass
            
            if current_score > best_score:
                best_score = current_score
                best_pts = _make_symmetric_points(current_params).copy()
        
        # Ring structure (5+5+3)
        if time.time() < deadline - 20.0:
            ring_inits = [
                [1.0, 0.6, 0.3, 0.0, np.pi/5.0, 0.0],
                [1.0, 0.7, 0.4, 0.0, 0.0, np.pi/6.0],
                [1.0, 0.55, 0.25, np.pi/10.0, np.pi/5.0, 0.0],
            ]
            for ring_init in ring_inits:
                if time.time() > deadline - 8.0:
                    break
                current_params = np.array(ring_init)
                current_score = _score(_make_ring_points(current_params))
                
                for alpha in [30.0, 100.0, 300.0]:
                    if time.time() > deadline - 6.0:
                        break
                    try:
                        res = minimize(
                            lambda p, a=alpha: _neg_ring_smooth_score(p, a),
                            current_params,
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12}
                        )
                        new_params = res.x
                        new_score = _score(_make_ring_points(new_params))
                        if new_score > current_score:
                            current_params = new_params
                            current_score = new_score
                    except Exception:
                        pass
                
                if time.time() < deadline - 4.0:
                    try:
                        res = minimize(
                            _neg_ring_score,
                            current_params,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13}
                        )
                        new_params = res.x
                        new_score = _score(_make_ring_points(new_params))
                        if new_score > current_score:
                            current_params = new_params
                            current_score = new_score
                    except Exception:
                        pass
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = _make_ring_points(current_params).copy()
    
    # === PHASE 5: Final high-precision polish ===
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            candidate_pts = res.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
