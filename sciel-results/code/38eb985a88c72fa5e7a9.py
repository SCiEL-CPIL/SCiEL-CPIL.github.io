import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

# Precompute all triangle index triples for vectorized area calculation
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


def _soft_min_objective(x_flat, beta=80.0, deadline=None):
    """
    Smooth surrogate for -min(triangle_area)/hull_area.
    Uses log-sum-exp to create a differentiable approximation of the min.
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    # Soft minimum approximation
    # We want to minimize the soft min, which is close to the hard min but smooth.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # Here we want to maximize min(ratios), so we minimize -soft_max(ratios) or similar.
    # Actually, let's use the standard soft min formula for the values we want to minimize.
    # We want to minimize the "worst case" ratio, i.e., min(ratios).
    # This is equivalent to maximizing -max(-ratios).
    # Or we can use the log-sum-exp trick on the negative values.
    
    # Let's stick to the form from the inspiration which seemed effective:
    # It computed a "soft_min_val" and returned its negative.
    # soft_min_val = r_min + log(sum(exp(-beta*(ratios - r_min)))) / beta
    # This is a lower bound on the min? No, log-sum-exp is a smooth approximation.
    # Let's verify: if one ratio is much smaller than others, exp(-beta*(r-r_min)) for the min is exp(0)=1, others are small.
    # log(1 + small) / beta is small. So soft_min_val ~ r_min.
    # If all ratios are equal to r, soft_min_val = r + log(n)*beta/beta = r + log(n). This shifts the value.
    # For optimization, the shift is constant, so the gradient points in the right direction.
    # We return -soft_min_val to maximize it.
    
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat, deadline=None):
    """Hard objective: -min_triangle_area / hull_area"""
    if deadline is not None and time.time() > deadline:
        return 1e10
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _normalize(points):
    """Normalize points to have hull area 1 and center at origin."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    return (points - center) * scale


def construct():
    start_time = time.time()
    # 120s limit, use 110s for safety
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    # Initial guess: Regular 13-gon
    init_points = _regular_polygon(N)
    init_points = _normalize(init_points)
    
    # Evaluate initial score
    init_score = _score(init_points)
    if init_score > best_score:
        best_score = init_score
        best_points = init_points.copy()
    
    # Optimization Strategy:
    # 1. Use the soft objective for the main optimization phase to allow smoother gradients.
    # 2. Use the hard objective for final polishing if time permits.
    # 3. Try multiple perturbations of the regular polygon.
    
    def run_optimization(initial_flat, objective_func, max_iter, deadline):
        try:
            result = minimize(
                objective_func,
                initial_flat,
                args=(deadline,),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            return result
        except Exception:
            return None

    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # Phase 1: Optimize from regular polygon using soft objective
    res = run_optimization(init_points.flatten(), _soft_min_objective, 500, deadline)
    if res is not None:
        new_points = res.x.reshape(N, 2)
        new_points = _normalize(new_points)
        new_score = _score(new_points)
        update_best(new_points, new_score)
        
        # If the soft optimization found a good point, try refining with hard objective
        if time.time() < deadline - 10:
            res2 = run_optimization(new_points.flatten(), _hard_objective, 200, deadline)
            if res2 is not None:
                new_points2 = res2.x.reshape(N, 2)
                new_points2 = _normalize(new_points2)
                new_score2 = _score(new_points2)
                update_best(new_points2, new_score2)

    # Phase 2: Try perturbed regular polygons
    rng = np.random.default_rng(42)
    for i in range(5):
        if time.time() > deadline - 15:
            break
        
        # Small noise to break symmetry
        noise = rng.normal(0, 0.05, size=(N, 2))
        perturbed_points = init_points + noise
        perturbed_points = _normalize(perturbed_points)
        
        res = run_optimization(perturbed_points.flatten(), _soft_min_objective, 300, deadline)
        if res is not None:
            new_points = res.x.reshape(N, 2)
            new_points = _normalize(new_points)
            new_score = _score(new_points)
            if update_best(new_points, new_score):
                # Refine with hard objective if improved
                if time.time() < deadline - 10:
                    res2 = run_optimization(new_points.flatten(), _hard_objective, 100, deadline)
                    if res2 is not None:
                        new_points2 = res2.x.reshape(N, 2)
                        new_points2 = _normalize(new_points2)
                        new_score2 = _score(new_points2)
                        update_best(new_points2, new_score2)

    # Phase 3: Final fine-tuning on the best point found
    if best_points is not None and time.time() < deadline - 5:
        best_flat = best_points.flatten()
        res = run_optimization(best_flat, _hard_objective, 500, deadline)
        if res is not None:
            new_points = res.x.reshape(N, 2)
            new_points = _normalize(new_points)
            new_score = _score(new_points)
            update_best(new_points, new_score)

    if best_points is None:
        return _regular_polygon(N)
        
    return best_points

# EVOLVE-BLOCK-END
