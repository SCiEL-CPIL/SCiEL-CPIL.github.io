"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    
    Applies a combined penalty strategy with smooth surrogates:
    1. Base penalty proportional to edge usage count (historical)
    2. Strong penalty on edges of the current local-optimal tour
    3. Usage-scaled amplification to break stagnant tours
    4. Smooth soft-min surrogate for neighborhood disruption
    
    Returns a modified distance matrix that biases 2-opt away from
    frequently-used and currently-present edges.
    """
    n = edge_distance.shape[0]
    
    # Work with a copy
    penalized = edge_distance.astype(np.float64).copy()
    
    # Compute reference scale: mean of the n smallest off-diagonal distances
    # This gives a representative "good edge" length for calibration
    flat = edge_distance.ravel()
    # Exclude diagonal (self-distances) - they're 0
    off_diag = flat[~np.eye(n, dtype=bool).ravel()]
    # Use the 50th percentile of off-diagonal distances as a robust scale
    scale = np.percentile(off_diag, 50) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    # --- Penalty 1: Historical usage penalty ---
    # Additive penalty scaled by usage count, with saturation
    max_usage = edge_n_used.max()
    if max_usage > 0:
        # Normalize usage to [0, 1]
        usage_norm = edge_n_used.astype(np.float64) / max_usage
        # Penalty: base + linear growth, capped
        usage_penalty = scale * (0.08 + 0.12 * usage_norm)
        penalized += usage_penalty
    
    # --- Penalty 2: Strong penalty on current tour edges ---
    # The current tour edges should be heavily penalized to force exploration
    tour_len = len(local_opt_tour)
    if tour_len >= 2:
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            # Apply a strong multiplicative penalty to these specific edges
            # Scale: make them ~2.5x their true distance
            # But also add an additive component based on usage
            usage_count = edge_n_used[u, v]
            # Stronger penalty for edges used more often
            mult = 2.0 + 0.3 * min(usage_count, 10) / 10.0
            penalized[u, v] *= mult
            penalized[v, u] *= mult
    
    # --- Penalty 3: Smooth neighborhood disruption using soft-min surrogate ---
    # Instead of hard argmin, use a smooth soft-min to perturb the 2nd/3rd nearest
    # neighbors, which provides better gradient behavior for 2-opt
    if tour_len >= 2:
        # Temperature parameter for soft-min (lower = sharper, higher = smoother)
        temp = 0.05 * scale
        
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            
            # Get all edges from u excluding self and the tour partner
            row_u = penalized[u].copy()
            row_u[u] = np.inf
            row_u[v] = np.inf  # exclude the direct tour partner
            
            if np.all(np.isinf(row_u)):
                continue
            
            # Soft-min: -log(sum(exp(-x/temp))) / log(n) gives a smooth approximation
            # of the minimum. We want to identify the 2nd and 3rd smallest.
            
            # Use log-sum-exp for numerical stability
            min_val = np.min(row_u[np.isfinite(row_u)])
            shifted = row_u - min_val
            shifted[~np.isfinite(shifted)] = 1e10
            
            # Soft-min gives a smooth lower bound; we use it to weight penalties
            log_sum_exp = np.log(np.sum(np.exp(-shifted / temp)))
            soft_min = min_val + temp * log_sum_exp
            
            # For the 2nd and 3rd neighbors, apply a penalty proportional to 
            # how close they are to the minimum (smooth surrogate)
            # Edges close to the minimum get less penalty, farther ones get more
            # This encourages 2-opt to consider alternatives
            
            finite_mask = np.isfinite(row_u)
            if np.any(finite_mask):
                # Normalize distances relative to the soft-min
                rel_dist = (row_u[finite_mask] - soft_min) / (scale + 1e-10)
                # Apply a smooth penalty that grows with relative distance
                # but is bounded to avoid extreme values
                penalty_weight = np.tanh(0.5 * rel_dist) * 0.03 * scale
                # Only apply to non-inf entries
                penalized[u, finite_mask] += penalty_weight
                penalized[finite_mask, u] += penalty_weight
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, 1e10)
    
    # Final polish: ensure the matrix is well-conditioned for 2-opt
    # Clip extremely large values to prevent numerical issues
    penalized = np.clip(penalized, 0.0, 1e6 * scale)
    
    return penalized
# EVOLVE-BLOCK-END
