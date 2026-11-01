"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
from scipy.spatial.distance import pdist

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    # Compute all pairwise distances
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
    
    def get_distance_vector(G):
        """Return vector of squared distances for all pairs from Gram matrix G."""
        diag = np.diag(G)
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            d_sq[k] = diag[i] + diag[j] - 2 * G[i, j]
        return d_sq

    def objective(G_flat, T, mu_psd, mu_rank, mu_scale):
        """
        Objective function for Gram matrix optimization.
        T: Temperature for smooth max/min approximation
        mu_psd: Weight for PSD penalty
        mu_rank: Weight for Rank penalty
        mu_scale: Weight for Scale penalty (Tr(G)=1)
        """
        G = G_flat.reshape(n, n)
        # Enforce symmetry
        G = (G + G.T) / 2
        
        # 1. Compute eigenvalues of G
        eigvals = np.linalg.eigvalsh(G)
        
        # PSD penalty: sum of negative eigenvalues squared
        neg_eigs = eigvals[eigvals < 0]
        psd_penalty = np.sum(neg_eigs ** 2) if len(neg_eigs) > 0 else 0.0
        
        # Rank penalty: sum of eigenvalues from index d to end squared
        # We want rank <= d, so eigenvalues d..n-1 should be 0
        # Note: eigvals are sorted ascending. So indices d-1 to n-1 are the large ones?
        # No, eigvalsh returns ascending.
        # We want the smallest d eigenvalues to be non-zero, and the rest to be 0?
        # No, for a rank-d PSD matrix, the largest d eigenvalues are non-zero, smallest n-d are 0.
        # So we penalize eigvals[0 : n-d] (the smallest ones) being non-zero?
        # Wait. If rank is d, then d eigenvalues are > 0 and n-d are 0.
        # Since they are sorted ascending, eigvals[0] ... eigvals[n-d-1] should be 0.
        # eigvals[n-d] ... eigvals[n-1] should be >= 0.
        # So rank penalty should be sum of squares of the smallest n-d eigenvalues.
        num_zero = n - d
        rank_penalty = np.sum(eigvals[:num_zero] ** 2) if num_zero > 0 else 0.0
        
        # 2. Compute squared distances
        d_sq = get_distance_vector(G)
        
        # Handle invalid distances
        if np.any(d_sq < -1e-6):
            return 1e10 + mu_psd * psd_penalty + mu_rank * rank_penalty
            
        # Clip small negative distances to 0 for stability
        d_sq_clipped = np.maximum(d_sq, 1e-12)
        
        # 3. Smooth Max and Min of squared distances
        # soft_max = T * log(sum(exp(d_sq / T)))
        # soft_min = -T * log(sum(exp(-d_sq / T)))
        
        # To avoid overflow, subtract max/min before exp
        max_d_sq = np.max(d_sq_clipped)
        min_d_sq = np.min(d_sq_clipped)
        
        # Soft max
        exp_max = np.exp((d_sq_clipped - max_d_sq) / T)
        soft_max = max_d_sq + T * np.log(np.sum(exp_max))
        
        # Soft min
        exp_min = np.exp((min_d_sq - d_sq_clipped) / T)
        soft_min = min_d_sq - T * np.log(np.sum(exp_min))
        
        if soft_min < 1e-12:
            return 1e10 + mu_psd * psd_penalty + mu_rank * rank_penalty
            
        # Objective: log(soft_max) - log(soft_min) approximates log(max/min)
        # We want to minimize (max/min)^2, which is 2 * log(max/min)
        # So minimizing log(soft_max) - log(soft_min) is equivalent up to a factor.
        
        log_ratio = np.log(soft_max) - np.log(soft_min)
        
        # 4. Scale penalty: Tr(G) = 1
        trace_G = np.trace(G)
        scale_penalty = (trace_G - 1.0) ** 2
        
        # Total objective
        return log_ratio + mu_psd * psd_penalty + mu_rank * rank_penalty + mu_scale * scale_penalty

    # Initializations
    np.random.seed(42)
    
    # Generate multiple initial candidates
    # 1. Random rank-d matrix G = V V^T
    # We want Tr(G) ~ 1, so scale V appropriately.
    V0 = np.random.randn(n, d)
    G0 = V0 @ V0.T
    # Normalize G0 to have trace 1
    tr0 = np.trace(G0)
    if tr0 > 1e-10:
        G0 /= tr0
    G0_flat = G0.flatten()
    
    # 2. Another random initialization
    V1 = np.random.randn(n, d)
    G1 = V1 @ V1.T
    tr1 = np.trace(G1)
    if tr1 > 1e-10:
        G1 /= tr1
    G1_flat = G1.flatten()
    
    # 3. A more structured initialization: Random points on a sphere?
    # Project random points to rank 3?
    # Just use random again for now.
    
    best_x = None
    best_val = 1e10
    
    # Annealing schedule for Temperature T
    # Start with high T to smooth the landscape, then decrease T
    T_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    
    # Weights for penalties
    # These need to be strong enough to enforce constraints but not dominate the objective
    mu_psd = 100.0
    mu_rank = 100.0
    mu_scale = 10.0
    
    for x0 in [G0_flat, G1_flat]:
        current_x = x0
        for T in T_schedule:
            try:
                # Define the objective for this specific T
                def obj_func(x, T_val=T, mp=mu_psd, mr=mu_rank, ms=mu_scale):
                    return objective(x, T_val, mp, mr, ms)
                
                res = minimize(obj_func, current_x, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
                
                # Update current_x for the next temperature step
                # Only update if the optimization made progress or is valid
                if res.success or res.fun < 1e9:
                    current_x = res.x
                else:
                    # If it failed badly, keep the previous best
                    pass
                    
            except Exception as e:
                # If an exception occurs, break the annealing for this start
                break

    if best_x is None:
        # Fallback to random points
        return np.random.randn(n, d)
    
    G_opt = best_x.reshape(n, n)
    G_opt = (G_opt + G_opt.T) / 2
    
    # Extract points from Gram matrix
    # We need to find V such that V V^T = G_opt.
    # Perform eigen-decomposition and keep top d eigenvectors.
    eigvals, eigvecs = np.linalg.eigh(G_opt)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top d eigenvectors
    # Ensure eigenvalues are non-negative
    eigvals_clipped = np.maximum(eigvals[:d], 0)
    V = eigvecs[:, :d] * np.sqrt(eigvals_clipped)
    
    points = V
    
    # Post-processing: Refine coordinates directly
    # This helps clean up any numerical drift from the Gram matrix optimization
    
    def objective_coords(x):
        V = x.reshape(n, d)
        d_sq = pdist(V) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
        T = 0.01 # Low temperature for sharp max/min
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        exp_max = np.exp((d_sq - max_d_sq) / T)
        soft_max = max_d_sq + T * np.log(np.sum(exp_max))
        
        exp_min = np.exp((min_d_sq - d_sq) / T)
        soft_min = min_d_sq - T * np.log(np.sum(exp_min))
        
        if soft_min < 1e-12:
            return 1e10
            
        return np.log(soft_max) - np.log(soft_min)

    try:
        res_final = minimize(objective_coords, points.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        # Check if the refined points are better
        score_before = _compute_score(points)
        score_after = _compute_score(res_final.x.reshape(n, d))
        
        if score_after < score_before:
            points = res_final.x.reshape(n, d)
    except Exception:
        pass

    return points

# EVOLVE-BLOCK-END
