# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    # Helper to compute the exact score R
    def exact_score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Helper for smooth min/max using log-sum-exp
    # We want to maximize (d_min / d_max)^2
    # Let's minimize -log(d_min/d_max) = log(d_max) - log(d_min)
    # Or just maximize the ratio directly with smooth approximations.
    
    def smooth_objective(flat_pts, tau):
        """
        Minimize negative log ratio.
        Returns (val, grad) if jac is requested, but we will use analytic or numeric.
        Actually, scipy L-BFGS-B can use numerical gradient, but analytic is faster.
        Given the complexity, let's try numeric first for simplicity, or implement analytic.
        For 32 dims, numeric is okay if function evals are cheap.
        """
        pts = flat_pts.reshape(n, d)
        dists = pdist(pts)
        
        # Soft Min: -tau * log(sum(exp(-d/tau)))
        # Soft Max: tau * log(sum(exp(d/tau)))
        
        # Numerical stability:
        # max_d = np.max(dists)
        # min_d = np.min(dists)
        
        # Log-sum-exp for max:
        m_max = np.max(dists)
        log_sum_exp_max = m_max + np.log(np.sum(np.exp((dists - m_max) / tau)))
        d_max_soft = tau * log_sum_exp_max
        
        # Log-sum-exp for min:
        # min(d) = - max(-d)
        m_min = np.min(dists)
        log_sum_exp_min_neg = m_min + np.log(np.sum(np.exp((dists - m_min) / tau))) # This is log sum exp of d, not -d
        # We need log sum exp of -d
        # max(-d) = -min(d)
        # Let's do it properly.
        
        # d_min_soft = -tau * log(sum(exp(-d/tau)))
        # Let a = -d/tau. max(a) = -min(d)/tau.
        # sum(exp(a)) = exp(max(a)) * sum(exp(a - max(a)))
        # log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))
        # d_min_soft = -tau * ( -min(d)/tau + log(sum(exp((-d + min(d))/tau))) )
        #             = min(d) - tau * log(sum(exp((min(d) - d)/tau)))
        
        # This is getting complicated. Let's use a simpler surrogate:
        # Maximize d_min - lambda * (d_max - 1) is not scale invariant.
        
        # Let's stick to maximizing ratio.
        # Objective to minimize: -log(d_min_soft / d_max_soft) = log(d_max_soft) - log(d_min_soft)
        
        # d_max_soft approx = tau * log(sum(exp(d/tau)))
        # d_min_soft approx = -tau * log(sum(exp(-d/tau)))
        
        # log(d_max_soft) = log(tau) + log(log(sum(exp(d/tau))))
        # log(d_min_soft) = log(-tau) + log(-log(sum(exp(-d/tau)))) -> This is bad, log of log.
        
        # Better: Minimize (d_max_soft / d_min_soft)
        # Or Maximize (d_min_soft / d_max_soft)
        
        # Let's use:
        # val = d_min_soft / d_max_soft
        # We want to maximize val.
        # So minimize -val.
        
        # d_max_soft:
        # To avoid overflow, shift by max.
        d_max_shift = np.max(dists)
        sum_exp_max = np.sum(np.exp((dists - d_max_shift) / tau))
        d_max_soft = d_max_shift + tau * np.log(sum_exp_max)
        
        # d_min_soft:
        d_min_shift = np.min(dists)
        sum_exp_min = np.sum(np.exp((d_min_shift - dists) / tau))
        d_min_soft = d_min_shift - tau * np.log(sum_exp_min)
        
        if d_max_soft == 0:
            return 0.0
        return - (d_min_soft / d_max_soft)

    def smooth_objective_grad(flat_pts, tau):
        # Analytic gradient is complex. We will rely on finite differences if needed,
        # or use 'COBYLA' which doesn't need gradients but is slower.
        # L-BFGS-B with finite diff is okay for 32 vars.
        return smooth_objective(flat_pts, tau)

    # Generate initial points
    inits = []
    
    # 1. Hexagonal lattice based
    # Generate a hexagonal grid of points
    # Center at 0,0.
    # Ring 0: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Total 19 points.
    
    hex_points = []
    # Generate coordinates for hex lattice
    # (i, j) -> (i + j/2, j * sqrt(3)/2)
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i + j / 2.0
            y = j * np.sqrt(3) / 2.0
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    
    # Find the 19 points closest to origin
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    top_19 = hex_points[sorted_indices[:19]]
    
    # We need to remove 3 points to get 16.
    # Try removing different combinations.
    # To keep it simple, try removing the 3 farthest, or 3 random, or symmetric.
    # Let's add a few variations.
    inits.append(top_19[:16]) # Just take first 16 (arbitrary order)
    
    # Remove 3 specific indices to break symmetry
    # Indices in top_19 corresponding to farthest points
    # Let's just add the top 19 with 3 removed in a few patterns
    for mask in [
        [0, 1, 2], # Remove first 3
        [16, 17, 18], # Remove last 3
        [0, 8, 16] # Symmetric-ish
    ]:
        remaining = [i for i in range(19) if i not in mask]
        inits.append(top_19[remaining])

    # 2. Random initializations
    np.random.seed(42)
    for _ in range(10):
        pts = np.random.randn(n, d)
        # Normalize to avoid extreme scales
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 2.0
        inits.append(pts)

    best_pts = None
    best_score = -1.0

    # Annealing parameters
    # Start with large tau, end with small tau
    taus = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01]
    
    # Optimization loop
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        current_pts = init.copy()
        
        # Run L-BFGS-B with annealing
        for tau in taus:
            if time.time() > deadline - 5:
                break
                
            # Flatten for scipy
            flat = current_pts.flatten()
            
            # Objective function wrapper
            def obj(x):
                return smooth_objective(x, tau)
                
            try:
                res = minimize(obj, flat, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success or res.status == 0:
                    current_pts = res.x.reshape(n, d)
            except Exception:
                pass
            
            # Check time
            if time.time() > deadline - 5:
                break
        
        # Evaluate exact score
        score = exact_score(current_pts)
        if score > best_score:
            best_score = score
            best_pts = current_pts.copy()

    # Final polish: Use Nelder-Mead on the exact objective for the best candidate
    if best_pts is not None and time.time() < deadline - 10:
        def exact_obj_neg(flat):
            pts = flat.reshape(n, d)
            return -exact_score(pts)
            
        flat_best = best_pts.flatten()
        try:
            res = minimize(exact_obj_neg, flat_best, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success:
                polished_pts = res.x.reshape(n, d)
                polished_score = exact_score(polished_pts)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts
        except Exception:
            pass

    # If best_pts is still None (shouldn't happen), fallback to random
    if best_pts is None:
        np.random.seed(42)
        best_pts = np.random.randn(n, d)

    return best_pts

# EVOLVE-BLOCK-END
