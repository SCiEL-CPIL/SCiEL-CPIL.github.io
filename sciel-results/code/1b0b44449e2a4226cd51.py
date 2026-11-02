import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0

    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area < 1e-12:
        return 0.0

    # Precompute triangle indices
    # Using combinations for clarity and correctness
    import itertools
    # For speed, we can vectorize, but N=14 is small enough that itertools might be comparable or we can use the precomputed indices
    # Let's use the vectorized approach with precomputed indices for speed
    # But we need to generate them. Let's do it once outside if possible? No, function is stateless.
    # Generating indices every time is cheap for N=14.
    
    # Fast vectorized min triangle area
    # Indices for i < j < k
    # We can use np.triu_indices or similar, but explicit is safer.
    # Let's use a static cache for indices
    if not hasattr(_score, '_indices'):
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _score._indices = (i[mask], j[mask], k[mask])
    
    i, j, k = _score._indices
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(flat_points):
    points = flat_points.reshape((N, 2))
    score = _score(points)
    if score < 1e-10:
        return 1e10
    return -score

def _soft_objective(flat_points, temperature=0.1):
    """
    Soft objective to help escape local minima by smoothing the min function.
    min(f) approx -T * log(sum(exp(-f_i/T)))
    """
    points = flat_points.reshape((N, 2))
    # Compute areas and hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10

    if hull_area < 1e-12:
        return 1e10

    i, j, k = _score._indices
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    normalized_areas = areas / hull_area
    
    # Soft min
    # To avoid overflow, subtract max
    max_val = np.max(normalized_areas)
    exp_terms = np.exp(-(normalized_areas - max_val) / temperature)
    soft_min = max_val + temperature * np.log(np.sum(exp_terms))
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_points = None
    best_score = -np.inf

    # Helper to update best
    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(initial_points)

    # 2. Optimized Regular 14-gon (Nelder-Mead)
    # This usually converges to a highly symmetric optimal configuration
    # We use a robust Nelder-Mead with adaptive simplex
    x0 = initial_points.flatten()
    
    # Run Nelder-Mead
    # We can try to run this for a while
    try:
        # Use a tighter tolerance and more iterations if time permits
        # But Nelder-Mead on 28D is slow.
        # Let's try to optimize the regular polygon by adjusting radii/angles slightly?
        # Or just run NM on the full space.
        
        # To make NM more effective, we can use a 'soft' objective initially to smooth the landscape
        # Then switch to hard objective.
        
        # Phase 1: Soft objective
        if time.time() < deadline - 20:
            res_soft = minimize(
                _soft_objective,
                x0,
                args=(0.05,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res_soft.success or (res_soft.fun < _objective(x0)):
                update_best(res_soft.x.reshape((N, 2)))
                x0 = res_soft.x
        
        # Phase 2: Hard objective refinement
        if time.time() < deadline - 5:
            res_hard = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_hard.success or (res_hard.fun < _objective(x0)):
                update_best(res_hard.x.reshape((N, 2)))
                
    except Exception:
        pass

    # 3. Try other symmetric configurations
    # Two concentric circles: 7 outer, 7 inner
    # Or 6 outer, 8 inner?
    # Or 14 points on a circle with alternating radii?
    
    # Let's try a "squashed" regular polygon
    # This might help if the optimal shape is not a regular polygon
    # But for max min triangle area, regular polygon is a strong candidate.
    
    # Let's try a few random restarts with NM if time allows
    rng = np.random.default_rng(42)
    
    # We can try to perturb the best points and re-optimize
    # Or try different initial seeds
    
    # Seed: Regular polygon with slight radial perturbation
    for r_factor in [0.9, 1.1]:
        if time.time() > deadline - 10:
            break
        
        pts = initial_points * r_factor
        # Random angular jitter
        angles_jitter = rng.uniform(-0.1, 0.1, N)
        pts[:, 0] = np.cos(angles + angles_jitter)
        pts[:, 1] = np.sin(angles + angles_jitter)
        
        update_best(pts)
        
        x0 = pts.flatten()
        try:
            res = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            update_best(res.x.reshape((N, 2)))
        except Exception:
            pass

    # One final check: ensure we return a valid array
    if best_points is None:
        return initial_points

    return best_points

# Pre-compute indices to avoid doing it on first call (slight speedup)
i, j, k = np.indices((N, N, N))
mask = (i < j) & (j < k)
_score._indices = (i[mask], j[mask], k[mask])
