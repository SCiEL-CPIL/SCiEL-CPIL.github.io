import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

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


def _affine_transform(points, a, b, c, d):
    """
    Apply affine transformation defined by 2x2 matrix [[a, b], [c, d]] to points.
    This maps (x, y) -> (a*x + b*y, c*x + d*y).
    """
    x = points[:, 0]
    y = points[:, 1]
    new_x = a * x + b * y
    new_y = c * x + d * y
    return np.column_stack([new_x, new_y])


def _optimize_affine(base_points, deadline):
    """
    Optimize the affine transformation of base_points to maximize the score.
    We optimize the 4 parameters of the affine matrix.
    Note: Translation and scaling are handled by normalization in the score,
    so we only need to find the best linear part (shear/rotation/scale ratios).
    """
    def objective(params):
        a, b, c, d = params
        transformed = _affine_transform(base_points, a, b, c, d)
        # Normalize to unit area and center
        normed = _normalize(transformed)
        score = _score(normed)
        if score <= 0:
            return 1e10
        return -score

    # Initial affine matrix: Identity
    init_params = np.array([1.0, 0.0, 0.0, 1.0])
    
    try:
        result = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result is not None:
            best_params = result.x
            best_transformed = _affine_transform(base_points, best_params[0], best_params[1], best_params[2], best_params[3])
            best_normed = _normalize(best_transformed)
            return best_normed, _score(best_normed)
    except Exception:
        pass
    
    return None, 0.0


def construct():
    start_time = time.time()
    # 120s limit, use 110s for safety
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

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

    # Strategy:
    # 1. Generate a set of diverse "base" configurations.
    #    a) Regular 13-gon
    #    b) 4x4 Grid (minus one corner) - as suggested by the directive
    #    c) Perturbed regular polygon
    # 2. For each base configuration, optimize the affine transformation (4 params)
    #    to find the best shear/scale/rotation that maximizes the min triangle area ratio.
    # 3. Take the best result from affine optimization.
    # 4. Perform a full 26-dimensional local search from the best affine result to fine-tune.

    base_configs = []

    # Config 1: Regular 13-gon
    reg_poly = _regular_polygon(N)
    base_configs.append(reg_poly)

    # Config 2: 4x4 Grid minus one corner
    # Create a 4x4 grid
    xs = np.linspace(-1.5, 1.5, 4)
    ys = np.linspace(-1.5, 1.5, 4)
    grid_points = []
    for x in xs:
        for y in ys:
            grid_points.append([x, y])
    grid_points = np.array(grid_points)
    # Remove one corner (e.g., last one) to get 15 points? No, we need 13.
    # 4x4 is 16 points. Remove 3 points?
    # Let's try removing 3 corners or edge points to make it more symmetric?
    # Or just take the first 13 points.
    # Better: A 3x3 grid (9 points) + 4 points on the boundary?
    # Let's stick to the directive: "centered 4x4 grid minus one corner" implies 15 points?
    # Wait, 4x4 is 16. Minus 1 is 15. We need 13.
    # Maybe "4x4 grid" means something else? Or just use a subset.
    # Let's create a 13-point grid-like structure.
    # 3x3 grid is 9. Add 4 midpoints of the outer square?
    grid_3x3 = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            grid_3x3.append([i, j])
    grid_3x3 = np.array(grid_3x3)
    
    # Add 4 points at the midpoints of the sides of the square [-1.5, 1.5]
    extra_points = np.array([
        [0, 1.5],
        [1.5, 0],
        [0, -1.5],
        [-1.5, 0]
    ])
    grid_config = np.vstack([grid_3x3, extra_points])
    base_configs.append(grid_config)

    # Config 3: Perturbed Regular Polygon
    rng = np.random.default_rng(42)
    perturbed = reg_poly + rng.normal(0, 0.1, size=(N, 2))
    base_configs.append(perturbed)

    # Phase 1: Affine Optimization for each base config
    for base in base_configs:
        if time.time() > deadline - 20:
            break
        
        # Normalize base config first to have reasonable scale
        base_norm = _normalize(base)
        
        aff_points, aff_score = _optimize_affine(base_norm, deadline)
        if aff_points is not None:
            update_best(aff_points, aff_score)

    # Phase 2: Full 26D optimization from the best point found so far
    if best_points is not None and time.time() < deadline - 10:
        # Try hard objective refinement
        res = run_optimization(best_points.flatten(), _hard_objective, 500, deadline)
        if res is not None:
            new_points = res.x.reshape(N, 2)
            new_points = _normalize(new_points)
            new_score = _score(new_points)
            update_best(new_points, new_score)

    # Phase 3: If time permits, try a few random full-space restarts
    if time.time() < deadline - 30:
        for i in range(3):
            if time.time() > deadline - 15:
                break
            
            # Random points in a square
            rand_points = rng.uniform(-1, 1, size=(N, 2))
            rand_points = _normalize(rand_points)
            
            res = run_optimization(rand_points.flatten(), _soft_min_objective, 200, deadline)
            if res is not None:
                new_points = res.x.reshape(N, 2)
                new_points = _normalize(new_points)
                new_score = _score(new_points)
                if update_best(new_points, new_score):
                    # Refine with hard objective
                    if time.time() < deadline - 10:
                        res2 = run_optimization(new_points.flatten(), _hard_objective, 200, deadline)
                        if res2 is not None:
                            new_points2 = res2.x.reshape(N, 2)
                            new_points2 = _normalize(new_points2)
                            new_score2 = _score(new_points2)
                            update_best(new_points2, new_score2)

    # Final fine-tuning
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
