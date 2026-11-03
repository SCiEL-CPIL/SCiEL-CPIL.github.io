"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import differential_evolution

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 1e-10:
        return 0.0
        
    # Calculate all triangle areas
    # We can vectorize this for speed
    # Indices for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_vals = i_idx[mask]
    j_vals = j_idx[mask]
    k_vals = k_idx[mask]
    
    p1 = points[i_vals]
    p2 = points[j_vals]
    p3 = points[k_vals]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _negative_score(x):
    """
    Objective function for differential_evolution.
    Minimizes negative score (i.e., maximizes score).
    """
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s

def _local_refine(points, max_time=10.0):
    """
    Perform a quick local refinement using coordinate descent or small random steps.
    This helps polish the result from the global optimizer.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    # Simple hill climbing with adaptive step size
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        # Try moving each point slightly
        for i in range(N):
            for _ in range(3): # a few random directions per point
                direction = rng.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                # Try positive and negative step
                for sign in [1, -1]:
                    new_points = current_points.copy()
                    new_points[i] += sign * step_size * direction
                    
                    new_score = _score(new_points)
                    if new_score > current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_points = current_points.copy()
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.7 # Reduce step size if stuck
            if step_size < 1e-6:
                break
        else:
            no_improve_count = 0
    
    return best_points, best_score

def construct():
    """
    Uses Differential Evolution to find a global optimum for the point configuration,
    followed by local refinement.
    """
    # Bounds for differential evolution
    # We can constrain points to a box, e.g., [-1, 1] x [-1, 1]
    # The normalization by hull area makes the scale invariant, so this is safe.
    bounds = [(-1.5, 1.5) for _ in range(2 * N)]
    
    # Start time
    start_time = time.time()
    
    # Run Differential Evolution
    # We limit maxiter and popsize to ensure it finishes in time.
    # N=14 -> 28 dimensions.
    # DE is stochastic, so we use a seed for reproducibility.
    try:
        result = differential_evolution(
            _negative_score, 
            bounds, 
            seed=42,
            maxiter=200, 
            popsize=15,
            tol=1e-4,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        best_points = result.x.reshape(-1, 2)
        best_score = -result.fun
        
    except Exception as e:
        # Fallback if DE fails or takes too long
        print(f"DE failed: {e}")
        # Use a simple random initialization + local refine
        rng = np.random.RandomState(42)
        best_points = rng.uniform(-1, 1, (N, 2))
        best_score = _score(best_points)
        
    # Check time remaining
    elapsed = time.time() - start_time
    time_remaining = 110.0 - elapsed # Leave 10s buffer for overhead
    
    if time_remaining > 5.0:
        # Local refinement
        refined_points, refined_score = _local_refine(best_points, max_time=time_remaining)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
            
    return best_points
# EVOLVE-BLOCK-END
