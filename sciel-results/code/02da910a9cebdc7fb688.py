import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3
PAIRS = list(itertools.combinations(range(N), 2))
NUM_PAIRS = len(PAIRS)

def _pairwise_sq_dist(X):
    """Compute squared distances for all pairs."""
    # X is (N, D)
    # diff[i, j, :] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    # Extract upper triangle
    iu = np.triu_indices(N, 1)
    return sq_dist[iu]

def _objective_smooth(X, tau):
    """
    Smoothed objective: log(max_sq_dist) - log(min_sq_dist)
    Using log-sum-exp for stability.
    """
    sq_dists = _pairwise_sq_dist(X)
    
    # Log max: log(sum(exp(tau * D))) / tau
    # To avoid overflow, subtract max
    max_d = np.max(sq_dists)
    min_d = np.min(sq_dists)
    
    # Soft max
    # L_max = (1/tau) * (log(sum(exp(tau * D))) )
    # = (1/tau) * (tau * max_d + log(sum(exp(tau * (D - max_d)))))
    # = max_d + (1/tau) * log(sum(exp(tau * (D - max_d))))
    exp_arg_max = tau * (sq_dists - max_d)
    log_sum_exp_max = max_d + (1.0 / tau) * np.log(np.sum(np.exp(exp_arg_max)))
    
    # Soft min: -log(sum(exp(-tau * D))) / tau
    # = -max_d_min + (1/tau) * log(sum(exp(-tau * (D - min_d)))) where max_d_min is min_d
    # Actually: log(min D) approx - (1/tau) log(sum(exp(-tau D)))
    # = - (1/tau) * (-tau * min_d + log(sum(exp(-tau * (D - min_d)))))
    # = min_d - (1/tau) * log(sum(exp(-tau * (D - min_d))))
    exp_arg_min = -tau * (sq_dists - min_d)
    log_sum_exp_min = min_d - (1/tau) * np.log(np.sum(np.exp(exp_arg_min)))
    
    # Objective: log(max) - log(min)
    # We want to minimize this.
    # Note: if tau is large, this approaches log(max_d) - log(min_d).
    return log_sum_exp_max - log_sum_exp_min

def _objective_exact(X):
    """
    Exact objective: (max_d / min_d)^2
    Since we are minimizing squared ratio, and ratio^2 = (max_d/min_d)^2.
    Wait, the task says "squared ratio of maximum to minimum pairwise distance".
    Ratio = d_max / d_min.
    Score = (d_max / d_min)^2 = d_max^2 / d_min^2.
    So we minimize max_sq_dist / min_sq_dist.
    """
    sq_dists = _pairwise_sq_dist(X)
    max_d = np.max(sq_dists)
    min_d = np.min(sq_dists)
    
    # Handle potential zero distance (distinct points constraint)
    if min_d < 1e-12:
        return 1e10
    return max_d / min_d

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_score = float('inf')
    best_X = None
    
    # Initial grid from seed, just in case it's better than random?
    # Seed score was 1.0 (which corresponds to raw value 2.045? No, seed score 1.0 means raw value is baseline.
    # The prompt says "score is seed/value". If seed program score is 1.0, then value_seed = value_seed.
    # Wait, "1.0 equals the seed program".
    # The seed program returns a grid.
    # Let's calculate the seed score to know the baseline raw value.
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    X_seed = np.array(grid, dtype=float)
    seed_raw = _objective_exact(X_seed)
    # print(f"Seed raw: {seed_raw}")
    
    # We want to beat seed_raw.
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Try a few structured starts (e.g. scaled to unit sphere)
    # 2. Random starts
    
    num_restarts = 50 # We will break early based on time
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial X
        if i == 0:
            # Use seed but scaled/rotated slightly?
            X = X_seed.copy()
            # Add small noise to break symmetry
            X += rng.normal(0, 0.1, X.shape)
        else:
            # Random points in a sphere of radius 1
            X = rng.normal(0, 1, (N, D))
            # Normalize to have similar scale
            X /= np.linalg.norm(X) * np.sqrt(N) # Rough normalization
        
        # Optimization loop with annealing tau
        current_X = X.copy()
        
        # Schedule of taus
        taus = [10, 50, 200, 1000, 5000]
        
        for tau in taus:
            if time.time() - start_time > time_limit:
                break
                
            # Define objective for this tau
            def obj_func(x_flat):
                x = x_flat.reshape(N, D)
                return _objective_smooth(x, tau)
            
            # Use BFGS or L-BFGS-B
            try:
                res = minimize(obj_func, current_X.flatten(), method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success:
                    current_X = res.x.reshape(N, D)
            except Exception:
                pass
        
        # Final polish with exact objective
        if time.time() - start_time > time_limit:
            break
            
        def obj_exact_func(x_flat):
            x = x_flat.reshape(N, D)
            return _objective_exact(x)
        
        try:
            # Powell is robust for non-smooth
            res = minimize(obj_exact_func, current_X.flatten(), method='Powell', 
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                current_X = res.x.reshape(N, D)
        except Exception:
            pass
        
        # Evaluate final score
        score = _objective_exact(current_X)
        
        if score < best_score:
            best_score = score
            best_X = current_X.copy()
            
            # If we beat the seed, we are good.
            # If we are significantly better, maybe we can stop early?
            # But let's try to improve further.
            
    # Ensure best_X is not None
    if best_X is None:
        return X_seed
    
    return best_X

if __name__ == "__main__":
    # Quick self-test
    X = construct()
    print("Score:", _objective_exact(X))
