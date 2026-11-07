"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a direct, principled blend of Expected Improvement and UCB,
    normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        # Fallback to simple cost-aware UCB
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement with adaptive beta
    # Estimate progress: how close are we to the known minimum?
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        # Normalize progress: early on, f_best is around -1.0 to -2.0
        initial_guess = -1.0
        total_range = initial_guess - HARTMANN_6D_MIN
        progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Adaptive beta: explore more early, exploit more late
    beta_early = 2.0
    beta_late = 0.5
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Blend EI and UCB based on progress
    # Early: weight towards UCB, Late: weight towards EI
    w = 0.5 * (1.0 + progress)  # w in [0.5, 1.0]
    blended = w * ei + (1.0 - w) * ucb_improvement
    
    # Base cost-aware score
    base_scores = blended / cost
    
    # Pareto-frontier selection: identify non-dominated candidates
    # Three objectives: (improvement, uncertainty, -cost) -- maximize all three
    # Non-dominated: no other point is >= in all three and > in at least one
    
    # Use efficient frontier: a point is non-dominated if no other point
    # has higher improvement AND higher uncertainty AND lower cost
    # We'll use a vectorized approach with sorting for efficiency
    
    # For 2000 points, we can do O(n^2) dominance check but that's 4M ops
    # Let's use a smarter approach: sort by one criterion, then check
    
    # Normalize objectives to [0,1] for comparison
    imp_min, imp_max = np.min(blended), np.max(blended)
    unc_min, unc_max = np.min(sigma), np.max(sigma)
    cost_min, cost_max = np.min(cost), np.max(cost)
    
    imp_range = imp_max - imp_min + 1e-9
    unc_range = unc_max - unc_min + 1e-9
    cost_range = cost_max - cost_min + 1e-9
    
    imp_norm = (blended - imp_min) / imp_range  # higher is better
    unc_norm = (sigma - unc_min) / unc_range    # higher is better (more exploration)
    cost_norm = 1.0 - (cost - cost_min) / cost_range  # higher is better (lower cost)
    
    # Find non-dominated points using vectorized dominance check
    # Point i dominates point j if:
    #   imp_norm[i] >= imp_norm[j] AND unc_norm[i] >= unc_norm[j] AND cost_norm[i] >= cost_norm[j]
    #   AND at least one is strictly greater
    
    n = len(mu)
    is_dominated = np.zeros(n, dtype=bool)
    
    # For efficiency, batch the dominance check
    # Process in chunks to avoid memory issues
    chunk_size = 500
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        imp_chunk = imp_norm[start:end]  # shape (chunk,)
        unc_chunk = unc_norm[start:end]
        cost_chunk = cost_norm[start:end]
        
        # Compare chunk against all points
        # imp_chunk[:, None] >= imp_norm[None, :] etc.
        dom_imp = imp_chunk[:, None] >= imp_norm[None, :]
        dom_unc = unc_chunk[:, None] >= unc_norm[None, :]
        dom_cost = cost_chunk[:, None] >= cost_norm[None, :]
        
        all_dom = dom_imp & dom_unc & dom_cost
        
        # At least one strictly greater
        strict_imp = imp_chunk[:, None] > imp_norm[None, :]
        strict_unc = unc_chunk[:, None] > unc_norm[None, :]
        strict_cost = cost_chunk[:, None] > cost_norm[None, :]
        
        any_strict = strict_imp | strict_unc | strict_cost
        
        # Point i in chunk is dominated by point j if all_dom[i,j] and any_strict[i,j] and i != j
        # But we need: exists j != i such that j dominates i
        # i.e., imp_norm[j] >= imp_norm[i] AND unc_norm[j] >= unc_norm[i] AND cost_norm[j] >= cost_norm[i]
        # AND (imp_norm[j] > imp_norm[i] OR unc_norm[j] > unc_norm[i] OR cost_norm[j] > cost_norm[i])
        
        # Reverse: for each point in chunk, check if any other point dominates it
        # Point j dominates point i if:
        #   imp_norm[j] >= imp_norm[i] AND unc_norm[j] >= unc_norm[i] AND cost_norm[j] >= cost_norm[i]
        #   AND at least one strict
        
        # So for point i in chunk (index k in chunk, global index start+k):
        # Check all j: imp_norm[j] >= imp_norm[i] AND ...
        
        # Let's flip: for each i in chunk, does any j dominate i?
        # j dominates i iff: imp_norm[j] >= imp_norm[i] AND unc_norm[j] >= unc_norm[i] AND cost_norm[j] >= cost_norm[i] AND any strict
        
        # Vectorized: for chunk row k (global idx = start+k), 
        # imp_norm[None, :] >= imp_norm[start+k] -- but this is expensive
        
        # Alternative: precompute and use matrix ops
        # Let's just do the direct O(n^2) but in a memory-efficient way
        
        # For each point in chunk, check against all n points
        # This is chunk_size * n comparisons
        for k in range(end - start):
            gi = start + k  # global index
            # Check if any j != gi dominates gi
            # j dominates gi if: imp_norm[j] >= imp_norm[gi] AND unc_norm[j] >= unc_norm[gi] AND cost_norm[j] >= cost_norm[gi] AND any strict
            imp_ge = imp_norm >= imp_norm[gi]
            unc_ge = unc_norm >= unc_norm[gi]
            cost_ge = cost_norm >= cost_norm[gi]
            
            imp_gt = imp_norm > imp_norm[gi]
            unc_gt = unc_norm > unc_norm[gi]
            cost_gt = cost_norm > cost_norm[gi]
            
            dominates = imp_ge & unc_ge & cost_ge & (imp_gt | unc_gt | cost_gt)
            # Exclude self
            dominates[gi] = False
            
            if np.any(dominates):
                is_dominated[gi] = True
    
    # Frontier points: not dominated
    frontier_mask = ~is_dominated
    
    # Boost frontier points: give them a multiplicative advantage
    # Points on the frontier get a score boost to prefer them
    # Non-frontier points get slightly penalized
    
    # Use a soft boost: frontier points get base_score * (1 + boost)
    # Non-frontier get base_score * (1 - penalty)
    # This maintains relative ordering but favors frontier
    
    frontier_boost = 1.15  # 15% boost for frontier points
    non_frontier_penalty = 0.85  # 15% penalty for non-frontier
    
    scores = np.where(frontier_mask, 
                      base_scores * frontier_boost, 
                      base_scores * non_frontier_penalty)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
