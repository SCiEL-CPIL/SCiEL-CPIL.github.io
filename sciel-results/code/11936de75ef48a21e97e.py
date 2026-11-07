import numpy as np

def acquisition(mu, sigma, f_best):
    n = len(mu)
    scores = np.zeros(n)
    
    if n == 0:
        return scores

    # Number of quantile bins
    k_bins = 20
    
    # Compute quantile edges for posterior mean
    quantiles = np.linspace(0, 1, k_bins + 1)
    bin_edges = np.quantile(mu, quantiles)
    
    # Assign bin indices
    bin_indices = np.digitize(mu, bin_edges[1:-1])
    bin_indices = np.clip(bin_indices, 0, k_bins - 1)
    
    # For each bin, select the point with maximum sigma (highest uncertainty)
    # This forces systematic exploration of high-uncertainty regions
    leaders = np.zeros(n, dtype=bool)
    
    for b in range(k_bins):
        mask = (bin_indices == b)
        if not np.any(mask):
            continue
        
        indices_in_bin = np.where(mask)[0]
        
        # Select point with highest sigma in this bin
        local_best_idx = np.argmax(sigma[indices_in_bin])
        global_best_idx = indices_in_bin[local_best_idx]
        
        # Score is based on sigma (higher uncertainty = higher priority)
        # We use -sigma as a proxy: we want to maximize the score, 
        # and higher sigma means more exploration value
        scores[global_best_idx] = sigma[global_best_idx]
        leaders[global_best_idx] = True
    
    # Set non-leaders to a very low value to force selection from leaders
    scores[~leaders] = -1e10
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
