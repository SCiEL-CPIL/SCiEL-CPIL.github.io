# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy: 
    1. Reframe the problem as maximizing the minimum distance subject to the maximum distance being 1.
    2. Use a smooth soft-min surrogate for the minimum distance.
    3. Use L-BFGS-B gradient-based optimization.
    4. Handle scale invariance by normalizing points in the objective function.
    5. Perform multiple random restarts to escape local minima.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def time_exceeded():
        return time.time() > deadline

    def get_score(points):
        """Calculate the actual score R = (d_min / d_max)^2."""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(points_flat, beta=100.0):
        """
        Objective function for minimization.
        We want to maximize the minimum distance, subject to the maximum distance being 1.
        
        Approach:
        1. Compute distances.
        2. Normalize points so that d_max = 1. This makes the problem scale-invariant 
           and effectively constrains d_max to 1.
        3. Compute soft-min of the normalized distances.
        4. Minimize -soft_min.
        
        Normalization:
        If d_max > 0, scale points by 1/d_max.
        Then the new d_max is 1, and we want to maximize the new d_min.
        The score is (d_min/d_max)^2 = (d_min_norm/1)^2 = d_min_norm^2.
        Maximizing d_min_norm is equivalent to maximizing the score.
        
        Soft min:
        soft_min(d) = -1/beta * log(sum(exp(-beta * d_i)))
        """
        points = points_flat.reshape(-1, 2)
        dists = pdist(points, 'euclidean')
        
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e10 # Penalty for degenerate points (all points coincident)
            
        # Normalize points such that d_max = 1
        # This effectively applies the constraint d_max <= 1 by scaling.
        # Since the objective is scale-invariant in the ratio, fixing the scale 
        # helps the optimizer converge to a specific shape rather than drifting.
        norm_points = points / d_max
        norm_dists = dists / d_max
        
        # We want to maximize the minimum of norm_dists.
        # Use soft-min to make it differentiable.
        
        # Clip to avoid numerical issues
        norm_dists = np.clip(norm_dists, 1e-6, 1.0)
        
        # Calculate soft min
        # Using log-sum-exp trick for stability:
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        # = min_x - 1/beta * log(sum(exp(-beta * (x - min_x))))
        
        min_val = np.min(norm_dists)
        shifted = norm_dists - min_val
        
        # Use a fixed beta for stability. 100 is a good trade-off between sharpness and numerical stability.
        b = 100.0
        vals = np.exp(-b * shifted)
        log_sum = np.log(np.sum(vals))
        soft_min_val = min_val - (1.0 / b) * log_sum
        
        # We want to maximize soft_min_val.
        # Minimize -soft_min_val.
        return -soft_min_val

    def run_optimization(init_points):
        """
        Run L-BFGS-B optimization from a given initial configuration.
        """
        x0 = init_points.flatten()
        
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_points = res.x.reshape(-1, 2)
        
        # Normalize to d_max = 1 for consistent output
        dists = pdist(best_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points / d_max
            
        return best_points

    # Initial configurations
    inits = []
    
    # 1. Grid-like init
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_points)
    
    # 2. Circle init
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_points = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(circle_points)
    
    # 3. Hexagonal-like init
    hex_points = []
    for row in range(-3, 4):
        for col in range(-3, 4):
            x = col + (row % 2) * 0.5
            y = row * 0.866
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    centers = np.argpartition(np.sum(hex_points**2, axis=1), 16)[:16]
    inits.append(hex_points[centers])
    
    # 4. Random inits
    np.random.seed(42)
    for seed in range(30):
        inits.append(np.random.uniform(-1, 1, (n, 2)))

    best_points = None
    best_score = 0.0

    # Run optimization on multiple inits
    for i, init in enumerate(inits):
        if time_exceeded():
            break
            
        # Run optimization
        # Slight perturbation to avoid exact local minima
        perturbed_init = init + np.random.randn(n, 2) * 0.05
        
        try:
            p = run_optimization(perturbed_init)
            s = get_score(p)
            
            if s > best_score:
                best_score = s
                best_points = p
        except Exception:
            pass

    # Final Polish: Take the best found and run a very high-precision optimization
    if best_points is not None and not time_exceeded():
        try:
            # Use a slightly perturbed version to avoid exact local minimum if it's a saddle
            p_refined = run_optimization(best_points + np.random.randn(n, 2) * 0.01)
            s_refined = get_score(p_refined)
            if s_refined > best_score:
                best_score = s_refined
                best_points = p_refined
        except Exception:
            pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)
        dists = pdist(best_points, 'euclidean')
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points / d_max

    return best_points

# EVOLVE-BLOCK-END
