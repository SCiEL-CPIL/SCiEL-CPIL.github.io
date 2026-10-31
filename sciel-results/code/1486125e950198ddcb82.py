import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull

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
        # Check for degeneracy: if all points are identical or nearly so
        if np.ptp(points, axis=0).max() < 1e-10:
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


def _objective(x_flat):
    """
    Objective function for differential_evolution.
    Returns negative score to maximize score.
    """
    points = x_flat.reshape(N, 2)
    # Penalize non-finite values
    if not np.all(np.isfinite(points)):
        return 1e10
    
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin against 120s limit
    
    # Define bounds for each coordinate. 
    # Since we normalize by hull area, the scale doesn't matter for the score, 
    # but it matters for the optimizer's step size. 
    # We can fix the domain to [0, 1] x [0, 1] or similar.
    # However, to avoid degeneracy where all points collapse, 
    # we rely on the penalty for hull_area ~ 0.
    # A good initial domain is [0, 1].
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Initial best: regular polygon
    # Center at 0.5, radius 0.5
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([0.5 + 0.49 * np.cos(angles), 0.5 + 0.49 * np.sin(angles)])
    best_score = _score(init_points)
    best_points = init_points.copy()
    
    if best_score <= 0:
        # Fallback if regular polygon fails (unlikely)
        best_points = np.random.rand(N, 2)
        best_score = _score(best_points)
        
    # Differential Evolution setup
    # maxiter: number of generations
    # popsize: population size factor
    # tol: tolerance for convergence
    # We need to be careful with time. 
    # DE can be slow. We will run it and check time periodically if possible, 
    # but scipy's DE doesn't have a built-in callback to stop early easily 
    # without custom loops. 
    # Instead, we set a reasonable maxiter and popsize that should finish in time.
    # 13 points -> 26 dims. 
    # popsize=15 -> 15*26 = 390 evaluations per generation.
    # If we run 50 generations, that's ~20,000 evaluations.
    # Each evaluation is fast (vectorized). 20k evals should take < 1s in numpy.
    # So we can afford more.
    
    # Let's try to run DE.
    try:
        # Use a seed for reproducibility if needed, but random is fine.
        # seed=42
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            # initial='latinhypercube' for better spread
            initial='latinhypercube'
        )
        
        # Check time
        if time.time() < deadline:
            candidate_points = res.x.reshape(N, 2)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
                
    except Exception:
        pass

    # If time remains, do a local polish on the best point found
    # using Nelder-Mead, which is good for fine-tuning.
    if time.time() < deadline - 10 and best_points is not None:
        try:
            from scipy.optimize import minimize
            
            def _hard_objective(x_flat):
                points = x_flat.reshape(N, 2)
                s = _score(points)
                if s <= 0:
                    return 1e10
                return -s
            
            # Polish
            res_polish = minimize(
                _hard_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if time.time() < deadline:
                polished_points = res_polish.x.reshape(N, 2)
                polished_score = _score(polished_points)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_points.copy()
                    
        except Exception:
            pass

    # Final check
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Return regular polygon as ultimate fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([0.5 + 0.49 * np.cos(angles), 0.5 + 0.49 * np.sin(angles)])
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
