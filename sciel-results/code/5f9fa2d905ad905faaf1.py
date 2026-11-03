"""Improved program for the suite task; uses smooth optimization with hexagonal init."""
import numpy as np
from scipy.optimize import minimize

N = 50


def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _optimize_from_start(start_centers, bounds):
    """Run the annealing optimization from a specific start."""
    def objective(params, k):
        centers = params.reshape(N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_terms = dists / 2.0
        wall_dists = 0.5 - np.abs(centers)
        all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
        all_terms = all_terms[np.isfinite(all_terms)]
        if len(all_terms) == 0:
            return 0.0
        m = np.min(all_terms)
        shifted = all_terms - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # Annealing schedule
    k_values = [10, 30, 100, 300, 1000]
    
    for k in k_values:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(k,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            centers = result.x.reshape(N, 2)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
            
    # Final refinement with Powell (no gradient, handles non-smoothness better locally)
    try:
        # Powell can be slow, but for N=50 and 100 dims, it might be okay if we limit iterations.
        # Actually, L-BFGS-B is usually better for high dimensions. Let's stick to L-BFGS-B with higher k.
        result = minimize(
            objective, 
            best_centers.reshape(-1), 
            args=(5000,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        centers = result.x.reshape(N, 2)
        true_r = _calculate_true_radius(centers)
        
        if true_r > best_score:
            best_score = true_r
            best_centers = centers
    except Exception:
        pass
        
    return best_centers, best_score


def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Hexagonal grid
    s = 0.15
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid
    s_sq = 1.0 / np.sqrt(N)
    # Adjust to fit in [-0.5, 0.5]
    # Approx 7x7 grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.01, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    best_centers = None
    best_score = -1.0
    
    for init in inits:
        centers, score = _optimize_from_start(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # Final safety clip
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
