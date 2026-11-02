import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def objective(x):
        # x is shape (N, 2)
        c = x.reshape(N, 2)
        
        # Pairwise distances
        # c[i] - c[j]
        diff = c[i_idx] - c[j_idx]
        d_pair = np.linalg.norm(diff, axis=1)
        r_pair = d_pair / 2.0
        
        # Boundary distances
        r_bound = 1.0 - np.linalg.norm(c, axis=1)
        
        # Combine into one vector
        # If any distance is negative (overlap or outside), the exp will be huge, 
        # but soft-min handles it. However, if r_pair is negative, e^{-beta * negative} is huge.
        # We want to maximize the min. If min is negative, we want to push it up.
        # The soft-min of a set containing negatives will be negative.
        # Minimizing -soft-min means maximizing the soft-min.
        
        # To avoid numerical issues with large exponents if distances are very negative,
        # we can clip the distances or just let it be. 
        # With beta ~ 50, if dist is -0.1, exp(5) ~ 150. If dist is -1, exp(50) is overflow.
        # So we should clip the minimum distance to a small negative value, e.g., -0.5.
        
        all_dists = np.concatenate([r_pair, r_bound])
        all_dists = np.maximum(all_dists, -0.5) # Prevent overflow
        
        beta = 50.0
        # Soft min
        # log-sum-exp trick for stability: max_val = max(all_dists)
        # softmin = max_val - (1/beta) * log(sum(exp(-beta * (all_dists - max_val))))
        # Wait, standard softmin is -1/beta * log(sum(exp(-beta * x)))
        # Let's use the stable version:
        # m = max(all_dists)
        # softmin = m - (1/beta) * log(sum(exp(-beta * (all_dists - m))))
        
        m = np.max(all_dists)
        # If m is very large, all_dists - m is <= 0. exp is safe.
        # If m is negative, same.
        # If all_dists contains -0.5, and m is 0.1, then -0.5 - 0.1 = -0.6. exp(-50 * -0.6) = exp(30) ~ 1e13. Safe.
        
        exp_term = np.exp(-beta * (all_dists - m))
        log_sum = np.log(np.sum(exp_term))
        soft_min_val = m - (1.0 / beta) * log_sum
        
        return -soft_min_val

    # Initial guess: Hexagonal lattice
    # The seed program uses a specific hex lattice. Let's generate a decent initial guess.
    # We can use the seed's logic to get an initial point, or just random.
    # Random points in the circle are a good start for global optimization if we have restarts.
    # But hex lattice is closer to the solution structure.
    
    # Let's use a few restarts with different initializations if time allows.
    # For N=50, one run might take 10-30 seconds.
    # We can do 3-5 restarts.
    
    best_x = None
    best_score = -np.inf
    
    # Initialization 1: Hexagonal
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    # The seed logic finds a grid that fits N points.
    # Let's just generate a hex grid and take the first N points sorted by radius.
    k = 10
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points, radius = points[radius <= 1.0], radius[radius <= 1.0]
    # Sort by radius to pick central ones
    points = points[np.argsort(radius, kind="stable")[:N]]
    
    x0 = points.flatten()
    
    # Optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < deadline:
            x_opt = res.x.reshape(N, 2)
            # Calculate actual score
            diff = x_opt[i_idx] - x_opt[j_idx]
            d_pair = np.linalg.norm(diff, axis=1)
            r_pair = d_pair / 2.0
            r_bound = 1.0 - np.linalg.norm(x_opt, axis=1)
            score = np.min(np.concatenate([r_pair, r_bound]))
            
            if score > best_score:
                best_score = score
                best_x = x_opt
    except Exception as e:
        pass

    # If time remains, try random restarts
    if time.time() < deadline - 20.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            # Random initialization
            r = np.sqrt(np.random.rand(N))
            theta = 2 * np.pi * np.random.rand(N)
            x0_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()
            
            try:
                res = minimize(objective, x0_rand, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                x_opt = res.x.reshape(N, 2)
                diff = x_opt[i_idx] - x_opt[j_idx]
                d_pair = np.linalg.norm(diff, axis=1)
                r_pair = d_pair / 2.0
                r_bound = 1.0 - np.linalg.norm(x_opt, axis=1)
                score = np.min(np.concatenate([r_pair, r_bound]))
                
                if score > best_score:
                    best_score = score
                    best_x = x_opt
            except Exception as e:
                pass

    if best_x is None:
        # Fallback to initial hex lattice if optimization failed
        best_x = points
    
    return best_x
