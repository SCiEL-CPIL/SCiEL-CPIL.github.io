# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _reconstruct_points(free_params):
    """
    Reconstruct 16 points from 28 free parameters.
    Points 0 and 1 are fixed at (0,0) and (1,0).
    """
    n = 16
    points = np.zeros((n, 2))
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2:] = free_params.reshape((n - 2, 2))
    return points


def _objective_for_minimize(free_params):
    """
    Objective function for local minimizers (Nelder-Mead/L-BFGS-B).
    Minimizes -log(score) to maximize score.
    """
    points = _reconstruct_points(free_params)
    score = _compute_score(points)
    if score <= 0:
        return 1e10
    return -np.log(score)


def _simulated_annealing(rng, n_iter=50000, t0=1.0, t1=1e-6, alpha=0.9999):
    """
    Run Simulated Annealing on the reduced 28-dim space.
    """
    n_free = 14
    dim = n_free * 2
    
    # Initialize with random points
    current_params = rng.uniform(-2, 2, size=dim)
    current_points = _reconstruct_points(current_params)
    current_score = _compute_score(current_points)
    
    best_params = current_params.copy()
    best_score = current_score
    best_points = current_points.copy()
    
    for i in range(n_iter):
        # Adaptive step size: start large, shrink over time
        # Using a simple geometric cooling schedule
        temp = t0 * (t1 / t0) ** (i / n_iter)
        
        # Generate neighbor
        # Step size scales with temperature and dimension
        step_size = temp * 0.5 
        neighbor_params = current_params + rng.normal(0, step_size, size=dim)
        
        neighbor_points = _reconstruct_points(neighbor_params)
        neighbor_score = _compute_score(neighbor_points)
        
        # Acceptance criterion
        if neighbor_score > current_score:
            current_params = neighbor_params
            current_points = neighbor_points
            current_score = neighbor_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                best_points = current_points.copy()
        else:
            # Metropolis criterion: accept worse with probability exp((new - old) / T)
            # We are maximizing score, so delta = neighbor - current (negative)
            # P = exp(delta / T)
            if temp > 1e-10:
                delta = neighbor_score - current_score
                # Prevent overflow/underflow issues
                if delta / temp > -700: 
                    prob = np.exp(delta / temp)
                else:
                    prob = 0.0
                if rng.random() < prob:
                    current_params = neighbor_params
                    current_points = neighbor_points
                    current_score = neighbor_score
            
    return best_points, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    best_points = None
    best_score = 0.0

    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # --- Phase 1: Simulated Annealing with Local Refinement ---
    # We run multiple independent SA runs. Each run ends with a local refinement step.
    
    # Number of SA runs to attempt
    n_runs = 10
    
    # Pre-define some good initial structures for SA starts to help convergence
    # 1. Grid-like
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Normalize grid to have dist(p0, p1) = 1? 
    # p0=(0,0), p1=(1,0). Grid has p0=(0,0), p1=(1,0) if we shift/scale.
    # Current grid: (0,0), (1,0), (2,0)... (0,1)...
    # p0=(0,0), p1=(1,0) is already there in the 4x4 grid.
    # So free params are just the rest.
    
    # Let's just use random starts for SA, but maybe bias them towards a reasonable scale.
    # Scale: typical distances are O(1).
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
            
        rng = np.random.default_rng(seed)
        
        try:
            # Run SA
            # n_iter depends on time budget. 
            # 50k iterations of SA in Python might take ~2-5s. 
            # 10 runs * 5s = 50s. Leaves plenty of time for refinement.
            sa_points, sa_score = _simulated_annealing(rng, n_iter=50000)
            
            # Local Refinement using Nelder-Mead on the reduced space
            # Nelder-Mead is robust to non-smoothness
            free_params = sa_points[2:].flatten()
            
            # Check time
            if time.time() > deadline - 10:
                update_best(sa_points, sa_score)
                continue
                
            result = minimize(
                _objective_for_minimize,
                free_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < 1e9:
                opt_free_params = result.x
                opt_points = _reconstruct_points(opt_free_params)
                opt_score = _compute_score(opt_points)
                update_best(opt_points, opt_score)
            else:
                update_best(sa_points, sa_score)
                
        except Exception:
            continue
            
    # --- Phase 2: Polish Best Found ---
    if best_points is not None and time.time() < deadline - 10:
        # One more polish on the absolute best
        best_free_params = best_points[2:].flatten()
        try:
            result = minimize(
                _objective_for_minimize,
                best_free_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < 1e9:
                opt_free_params = result.x
                opt_points = _reconstruct_points(opt_free_params)
                opt_score = _compute_score(opt_points)
                update_best(opt_points, opt_score)
        except Exception:
            pass

    # Fallback if no valid points were found
    if best_points is None:
        # Return a simple grid
        best_points = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)

    return best_points


# EVOLVE-BLOCK-END
