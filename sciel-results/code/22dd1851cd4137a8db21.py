import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _compute_score(points):
    """Compute the actual score R = (d_min/d_max)^2 for a set of points."""
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _surrogate_objective(x, n_points, power):
    """
    Smooth surrogate: minimize - (power mean of distances) / (max distance)
    Using a high power makes the power mean approximate the min distance.
    """
    n_free = n_points - 2
    free_points = x.reshape((n_free, 2))
    points = np.zeros((n_points, 2))
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2:] = free_points
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    
    # Power mean with high power approximates min
    # Use a stable computation
    d_min_approx = (np.mean(dists ** power)) ** (1.0 / power)
    
    ratio = d_min_approx / d_max
    return -(ratio ** 2)

def _local_refine(points, n_points, time_budget=5.0):
    """Local refinement using L-BFGS-B on the smooth surrogate."""
    start_time = time.time()
    n_free = n_points - 2
    
    # Flatten free points
    free_points = points[2:]
    x0 = free_points.flatten()
    
    bounds = [(-5.0, 5.0)] * (n_free * 2)
    
    best_x = x0
    best_score = _compute_score(points)
    
    # Try different power values for the surrogate
    for power in [20, 50, 100]:
        if time.time() - start_time > time_budget:
            break
            
        def obj(x):
            return _surrogate_objective(x, n_points, power)
        
        try:
            res = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                free_points_opt = res.x.reshape((n_free, 2))
                points_opt = np.zeros((n_points, 2))
                points_opt[0] = [0.0, 0.0]
                points_opt[1] = [1.0, 0.0]
                points_opt[2:] = free_points_opt
                
                score_opt = _compute_score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_x = res.x.copy()
        except Exception:
            continue
    
    # Reconstruct best points
    free_points_best = best_x.reshape((n_free, 2))
    points_best = np.zeros((n_points, 2))
    points_best[0] = [0.0, 0.0]
    points_best[1] = [1.0, 0.0]
    points_best[2:] = free_points_best
    
    return points_best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    time_limit = 600.0  # Seconds
    margin = 15.0  # Safety margin

    n_points = 16
    n_free = n_points - 2
    
    # Initial good configuration: hexagonal lattice based
    # This provides a strong starting point
    def _make_hex_initial(seed):
        rng = np.random.RandomState(seed)
        # Start with a hexagonal arrangement
        points = np.zeros((n_points, 2))
        points[0] = [0.0, 0.0]
        points[1] = [1.0, 0.0]
        
        # Add points in a hexagonal pattern with some randomness
        angles = rng.uniform(0, 2*np.pi, n_free)
        radii = rng.uniform(0.5, 2.5, n_free)
        for i in range(n_free):
            points[2 + i] = [radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i])]
        
        return points
    
    best_result = None
    best_score = -np.inf
    
    # Simulated Annealing approach
    def _sa_run(initial_points, max_steps=50000, t_start=1.0, t_end=0.001, seed=42):
        """Run simulated annealing on the points."""
        rng = np.random.RandomState(seed)
        n_free = n_points - 2
        points = initial_points.copy()
        current_score = _compute_score(points)
        
        best_points = points.copy()
        best_score_local = current_score
        
        # Temperature schedule
        t = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max_steps)
        
        step_size = 0.1
        
        for step in range(max_steps):
            # Check time
            if time.time() - start_time > time_limit - margin:
                break
            
            # Generate neighbor
            neighbor = points.copy()
            # Randomly perturb 1-3 free points
            n_perturb = rng.randint(1, 4)
            idx = rng.randint(2, n_points, size=n_perturb)
            noise = rng.randn(len(idx), 2) * step_size
            neighbor[idx] += noise
            
            # Keep within reasonable bounds
            neighbor = np.clip(neighbor, -10, 10)
            
            neighbor_score = _compute_score(neighbor)
            
            # Accept if better or with probability exp(-delta/T)
            delta = neighbor_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
                points = neighbor
                current_score = neighbor_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_points = points.copy()
            
            # Cool down
            t *= cooling_rate
            
            # Reduce step size occasionally
            if step % 1000 == 0:
                step_size *= 0.95
        
        return best_points, best_score_local
    
    # Multiple restarts with diverse initial conditions
    n_restarts = 10
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - margin:
            break
        
        # Create initial configuration
        initial_points = _make_hex_initial(seed=42 + restart)
        
        # Run SA
        sa_points, sa_score = _sa_run(initial_points, max_steps=30000, t_start=0.5, t_end=0.0005, seed=1000 + restart)
        
        # Local refinement
        refined_points, refined_score = _local_refine(sa_points, n_points, time_budget=3.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_result = refined_points
    
    # If no result found, use a fallback
    if best_result is None:
        best_result = _make_hex_initial(42)
        best_score = _compute_score(best_result)
    
    # Final local refinement with more time if available
    elapsed = time.time() - start_time
    if elapsed < time_limit - margin - 5:
        final_points, final_score = _local_refine(best_result, n_points, time_budget=10.0)
        if final_score > best_score:
            best_score = final_score
            best_result = final_points
    
    return best_result
