import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_objective(x_flat, sharpness=100.0):
    """Smooth approximation of -min_area/hull_area using log-sum-exp."""
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 1e10
    # Soft min: -log(sum(exp(-areas * sharpness))) / sharpness / hull_area
    # This is smooth and approximates min(areas)/hull_area as sharpness -> inf
    neg_scaled = -areas * sharpness
    max_neg = np.max(neg_scaled)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    soft_min = -log_sum / sharpness
    return -soft_min / hull_area


def _hard_objective(x_flat):
    """Hard objective: -score."""
    points = x_flat.reshape(N, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _build_symmetric_config():
    """Build a symmetric 13-point configuration: center + 6 inner + 6 outer."""
    center = np.array([[0.0, 0.0]])
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner = np.column_stack([0.6 * np.cos(inner_angles), 0.6 * np.sin(inner_angles)])
    outer_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    outer = np.column_stack([1.2 * np.cos(outer_angles), 1.2 * np.sin(outer_angles)])
    return np.vstack([center, inner, outer])


def _build_template_13_0():
    """13 points: 1 center + 5 ring1 + 7 ring2."""
    center = [[0.0, 0.0]]
    r1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring1 = np.column_stack([0.5 * np.cos(r1_angles), 0.5 * np.sin(r1_angles)])
    r2_angles = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False)
    ring2 = np.column_stack([1.0 * np.cos(r2_angles), 1.0 * np.sin(r2_angles)])
    return np.vstack([center, ring1, ring2])


def _build_template_7_6():
    """13 points: 7 outer heptagon + 6 inner hexagon."""
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    inner_angles = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
    return np.vstack([outer, inner])


def _build_template_6_7():
    """13 points: 6 outer hexagon + 7 inner heptagon."""
    outer_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inner = np.column_stack([0.55 * np.cos(inner_angles), 0.55 * np.sin(inner_angles)])
    return np.vstack([outer, inner])


def _build_template_5_4_4():
    """13 points: 5 outer + 4 mid + 4 inner."""
    outer_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    mid_angles = np.linspace(np.pi/5, 2 * np.pi + np.pi/5, 4, endpoint=False)
    mid = np.column_stack([0.7 * np.cos(mid_angles), 0.7 * np.sin(mid_angles)])
    inner_angles = np.linspace(np.pi/4, 2 * np.pi + np.pi/4, 4, endpoint=False)
    inner = np.column_stack([0.4 * np.cos(inner_angles), 0.4 * np.sin(inner_angles)])
    return np.vstack([outer, mid, inner])


def _build_template_8_5():
    """13 points: 8 outer octagon + 5 inner pentagon."""
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    return np.vstack([outer, inner])


def _annealed_soft_min_optimize(x0, deadline, max_iters=5000):
    """Simulated annealing with soft-min objective, then hard polish."""
    rng = np.random.default_rng(12345)
    x = x0.copy()
    best_x = x.copy()
    best_obj = _soft_min_objective(x, sharpness=100.0)
    
    # Annealing schedule
    T_start = 0.01
    T_end = 1e-6
    n_iters = max_iters
    
    for it in range(n_iters):
        if time.time() > deadline - 3:
            break
        
        # Temperature
        progress = it / n_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        step_size = 0.05 * (1.0 - progress) + 0.005
        delta = rng.normal(0, step_size, size=x.shape)
        x_new = x + delta
        
        # Evaluate
        obj_new = _soft_min_objective(x_new, sharpness=100.0)
        obj_diff = obj_new - best_obj
        
        if obj_diff < 0 or rng.random() < np.exp(-obj_diff / T):
            x = x_new
            if obj_new < best_obj:
                best_obj = obj_new
                best_x = x_new.copy()
    
    return best_x, -best_obj


def _coordinate_wise_local_search(points, deadline, max_rounds=50):
    """Coordinate-wise local search for fine-tuning."""
    rng = np.random.default_rng(67890)
    x = points.flatten().copy()
    best_obj = _hard_objective(x)
    best_x = x.copy()
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ss_idx, ss in enumerate(step_sizes):
        for _ in range(max_rounds):
            if time.time() > deadline - 2:
                return best_x.reshape(N, 2), -best_obj
            
            # Pick random coordinate
            dim = rng.integers(0, N * 2)
            direction = rng.choice([-1, 1])
            
            x_new = x.copy()
            x_new[dim] += direction * ss
            
            obj_new = _hard_objective(x_new)
            if obj_new < best_obj:
                best_obj = obj_new
                best_x = x_new.copy()
                x = x_new.copy()
    
    return best_x.reshape(N, 2), -best_obj


def _nelder_mead_polish(points, deadline, maxiter=300):
    """Final Nelder-Mead polish with hard objective."""
    if time.time() > deadline - 1:
        return points, _score(points)
    
    try:
        result = minimize(
            _hard_objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x.reshape(N, 2), _score(result.x.reshape(N, 2))
    except Exception:
        pass
    
    return points, _score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin against 150s limit
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        if not np.all(np.isfinite(points)):
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Regular polygon
    candidates.append(_regular_polygon(N, 1.0))
    
    # Symmetric configurations
    candidates.append(_build_symmetric_config())
    candidates.append(_build_template_13_0())
    candidates.append(_build_template_7_6())
    candidates.append(_build_template_6_7())
    candidates.append(_build_template_5_4_4())
    candidates.append(_build_template_8_5())
    
    # Random starts
    rng = np.random.default_rng(42)
    for seed in range(20):
        rng_local = np.random.default_rng(seed * 1000 + 42)
        pts = rng_local.uniform(-1.5, 1.5, (N, 2))
        candidates.append(pts)
    
    # Score all candidates
    scored_candidates = []
    for pts in candidates:
        if len(pts) != N:
            continue
        s = _score(pts)
        if s > 0:
            scored_candidates.append((s, pts))
    
    # Sort by score and take top-5 diverse seeds
    scored_candidates.sort(key=lambda x: -x[0])
    top_seeds = [pts for _, pts in scored_candidates[:5]]
    
    if not top_seeds:
        top_seeds = [_regular_polygon(N, 1.0)]
    
    # Phase 2: Annealed soft-min optimization from each top seed
    for seed_idx, seed_pts in enumerate(top_seeds):
        if time.time() > deadline - 30:
            break
        
        x0 = seed_pts.flatten()
        try:
            optimized_x, obj_val = _annealed_soft_min_optimize(
                x0, deadline, max_iters=3000
            )
            optimized_pts = optimized_x.reshape(N, 2)
            _update_best(optimized_pts)
        except Exception:
            pass
    
    # Phase 3: Coordinate-wise local search from best
    if best_points is not None and time.time() < deadline - 10:
        try:
            refined_pts, refined_score = _coordinate_wise_local_search(
                best_points, deadline, max_rounds=100
            )
            _update_best(refined_pts)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            polished_pts, polished_score = _nelder_mead_polish(
                best_points, deadline, maxiter=200
            )
            _update_best(polished_pts)
        except Exception:
            pass
    
    # Phase 5: Additional random restarts if time permits
    if time.time() < deadline - 15:
        for extra_seed in [123, 456, 789, 1011, 1314]:
            if time.time() > deadline - 5:
                break
            try:
                rng_extra = np.random.default_rng(extra_seed)
                pts = rng_extra.uniform(-1.5, 1.5, (N, 2))
                
                # Quick anneal
                x0 = pts.flatten()
                optimized_x, obj_val = _annealed_soft_min_optimize(
                    x0, deadline, max_iters=1500
                )
                optimized_pts = optimized_x.reshape(N, 2)
                _update_best(optimized_pts)
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
