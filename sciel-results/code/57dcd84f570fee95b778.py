import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    Uses a fast greedy proxy to tune a blend of aggregate, geometric mean, and
    scarcity-premium efficiencies.
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin
    
    eps = 1e-9
    n_items = 100
    n_constraints = 5
    capacity = 25.0
    
    # Pre-compute base metrics
    # Avoid division by zero
    safe_weights = weights + eps
    ratios = prizes / safe_weights  # Shape (5, 100)
    
    # Aggregate efficiency: sum(prizes) / sum(weights)
    total_prizes = prizes.sum(axis=0)
    total_weights = weights.sum(axis=0)
    H_agg = total_prizes / (total_weights + eps)
    
    # Geometric mean of ratios: (prod_j p_j/w_j)^(1/K)
    # log(geo) = mean(log(ratios))
    log_ratios = np.log(ratios)
    H_geo = np.exp(log_ratios.mean(axis=0))
    
    # Constraint-specific scarcity premium:
    # For each item, find the constraint where its weight is the largest fraction
    # of its total weight. Scale its prize-to-weight ratio by the inverse of the
    # remaining capacity fraction for that specific constraint.
    # This prioritizes items that offer high value relative to their most limiting resource.
    
    # Fraction of total weight in each constraint for each item
    weight_fractions = weights / (total_weights + eps)  # Shape (5, 100)
    
    # For each item, find the most binding constraint (largest weight fraction)
    most_binding_idx = np.argmax(weight_fractions, axis=0)  # Shape (100,)
    
    # Get the weight fraction for the most binding constraint
    most_binding_frac = weight_fractions[most_binding_idx, np.arange(n_items)]  # Shape (100,)
    
    # Get the prize-to-weight ratio for the most binding constraint
    most_binding_ratio = ratios[most_binding_idx, np.arange(n_items)]  # Shape (100,)
    
    # Scarcity premium: scale by 1/(1 - fraction) to penalize items that consume
    # a large fraction of their most limiting resource
    # Use a smooth approximation: 1/(1 - frac) can blow up, so use 1/(1 - frac + eps)
    scarcity_multiplier = 1.0 / (1.0 - most_binding_frac + eps)
    
    # H_scarcity: ratio of most binding constraint scaled by scarcity multiplier
    H_scarcity = most_binding_ratio * scarcity_multiplier
    
    # Normalize all components to mean 1
    H_agg_norm = H_agg / (H_agg.mean() + eps)
    H_geo_norm = H_geo / (H_geo.mean() + eps)
    H_scar_norm = H_scarcity / (H_scarcity.mean() + eps)
    
    # Ensure positivity
    H_agg_norm = np.maximum(H_agg_norm, eps)
    H_geo_norm = np.maximum(H_geo_norm, eps)
    H_scar_norm = np.maximum(H_scar_norm, eps)
    
    def greedy_proxy(eta):
        """
        Evaluate a heuristic by simulating a greedy selection.
        This is a fast, deterministic proxy for how well the heuristic
        guides the ACO ants.
        """
        # Sort items by eta descending
        order = np.argsort(-eta)
        used_w = np.zeros(n_constraints)
        total_p = 0.0
        
        for idx in order:
            # Check if item fits
            w_item = weights[:, idx]
            if np.all(used_w + w_item <= capacity):
                used_w += w_item
                total_p += prizes[:, idx].sum()
            else:
                # In a true greedy, we might skip and check next.
                # In ACO, if an item doesn't fit, it's just not chosen.
                # This proxy assumes we only add if it fits immediately.
                pass
                
        return total_p

    best_eta = None
    best_score = -1.0
    
    # Grid search over alpha (weight of H_agg) and beta (weight of H_scarcity)
    # H = H_agg^alpha * H_geo^(1-alpha-beta) * H_scar^beta
    # where alpha + beta <= 1
    
    # Try a grid of (alpha, beta) pairs
    alphas = np.linspace(0.0, 1.0, 11)
    betas = np.linspace(0.0, 1.0, 11)
    
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
        for beta in betas:
            if alpha + beta > 1.0 + 1e-6:
                continue
            if time.time() > deadline - 10:
                break
                
            # Blend: Geometric interpolation
            # H = H_agg^alpha * H_geo^(1-alpha-beta) * H_scar^beta
            H_blend = (np.power(H_agg_norm, alpha) * 
                      np.power(H_geo_norm, 1.0 - alpha - beta) * 
                      np.power(H_scar_norm, beta))
            
            # Normalize to mean 1
            H_blend_norm = H_blend / (H_blend.mean() + eps)
            H_blend_norm = np.maximum(H_blend_norm, eps)
            H_blend_norm = np.where(np.isfinite(H_blend_norm), H_blend_norm, eps)
            
            score = greedy_proxy(H_blend_norm)
            
            if score > best_score:
                best_score = score
                best_eta = H_blend_norm.copy()
                
    # If the grid search didn't find anything (should not happen), fallback to H_geo
    if best_eta is None:
        best_eta = H_geo_norm.copy()
        best_eta = np.maximum(best_eta, eps)
        best_eta = np.where(np.isfinite(best_eta), best_eta, eps)
        
    # Final safety check
    if not np.all(np.isfinite(best_eta)):
        best_eta = np.ones(n_items)
        
    return best_eta
