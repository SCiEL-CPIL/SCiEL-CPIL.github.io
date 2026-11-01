"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
import itertools
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # Objective function: Minimize the smooth approximation of the ratio
    # We use log-sum-exp for max and min to ensure differentiability.
    # max(d) ~ T * log(sum(exp(d/T)))
    # min(d) ~ -T * log(sum(exp(-d/T)))
    # Ratio ~ max(d) / min(d)
    # We minimize this ratio.
    
    def objective(x):
        V = x.reshape(n, d)
        
        # Compute squared distances
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = V[i] - V[j]
            d_sq[k] = np.dot(diff, diff)
        
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        # Adaptive temperature T to handle scale invariance
        mean_d_sq = np.mean(d_sq)
        if mean_d_sq < 1e-10:
            return 1e10
        T = 0.1 * mean_d_sq
        
        # Stable log-sum-exp
        d_scaled = d_sq / T
        max_d_scaled = np.max(d_scaled)
        min_d_scaled = np.min(d_scaled)
        
        sum_exp_pos = np.sum(np.exp(d_scaled - max_d_scaled))
        max_approx = T * (max_d_scaled + np.log(sum_exp_pos))
        
        sum_exp_neg = np.sum(np.exp(-d_scaled + min_d_scaled))
        min_approx = -T * (-min_d_scaled + np.log(sum_exp_neg))
        
        if min_approx < 1e-12:
            return 1e10
            
        return max_approx / min_approx

    # Initializations
    np.random.seed(42)
    
    # 1. Random
    x0_1 = np.random.randn(n * d)
    
    # 2. Fibonacci Sphere (scaled)
    x0_2 = np.zeros(n * d)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        x0_2[i*d:(i+1)*d] = np.array([x, y, z])
    # Scale to have unit variance
    x0_2 /= np.std(x0_2)
    
    # 3. Regular Grid
    side = int(np.ceil(n ** (1.0 / d)))
    grid = list(itertools.product(range(side), repeat=d))[:n]
    x0_3 = np.array(grid, dtype=float).flatten()
    # Center and scale
    x0_3 -= np.mean(x0_3)
    x0_3 /= np.std(x0_3)

    best_x = None
    best_val = 1e10
    
    # Use Basin Hopping for global search
    # This is significantly better for non-convex geometric optimization
    
    for x0 in [x0_1, x0_2, x0_3]:
        try:
            # Basin hopping parameters
            # T is the temperature for the Metropolis acceptance criterion
            # stepsize is the scale of the random kicks
            res = basinhopping(objective, x0, 
                               niter=50, 
                               T=1.0, 
                               stepsize=0.5,
                               minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 500, 'ftol': 1e-8}},
                               disp=False)
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # If basin hopping didn't find anything good (or failed), fall back to local optimization from best init
    if best_x is None or best_val > 100:
         # Try local optimization from the inits
         for x0 in [x0_1, x0_2, x0_3]:
             try:
                 res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000})
                 if res.fun < best_val:
                     best_val = res.fun
                     best_x = res.x
             except Exception:
                 pass

    if best_x is None:
        # Fallback
        return np.array(grid, dtype=float)

    points = best_x.reshape(n, d)
    
    # Final local refinement with Nelder-Mead for higher precision
    try:
        res_final = minimize(objective, best_x, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res_final.fun < best_val:
            best_x = res_final.x
            best_val = res_final.fun
    except Exception:
        pass
    
    points = best_x.reshape(n, d)
    return points

# EVOLVE-BLOCK-END
