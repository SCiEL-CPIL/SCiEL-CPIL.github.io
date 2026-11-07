import numpy as np
import scipy.stats
import time

def heuristics(prizes, weights):
    start_time = time.time()
    # 5 seconds is a safe upper bound for this computation
    deadline = start_time + 5.0
    
    # Convert to numpy arrays to ensure shape and type
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Expected shape: (5, 100)
    C, N = prizes.shape
    
    eps = 1e-9
    
    # Calculate efficiency for each constraint
    # eff[c, i] = prizes[c, i] / (weights[c, i] + eps)
    # We use a small epsilon to avoid division by zero
    efficiencies = np.zeros_like(prizes)
    for c in range(C):
        efficiencies[c, :] = prizes[c, :] / (weights[c, :] + eps)
        
    # Calculate ranks for each constraint
    # We want rank 1 to be the best (highest efficiency)
    ranks = np.zeros_like(efficiencies)
    for c in range(C):
        # rankdata returns 1 for the smallest value
        # We want 1 for the largest value
        # So we rank the negative efficiencies, or invert the ranks
        r = scipy.stats.rankdata(efficiencies[c, :], method='min')
        # r is in [1, N]
        # Invert: best (largest eff, highest r) should become 1
        # worst (smallest eff, lowest r) should become N
        ranks[c, :] = N + 1 - r
        
    # Normalize ranks to [0, 1]
    # Rank 1 -> 0
    # Rank N -> 1
    if N > 1:
        norm_ranks = (ranks - 1) / (N - 1)
    else:
        norm_ranks = np.zeros_like(ranks)
        
    # Attractiveness component: 1 - normalized rank
    # Best item (rank 1) -> 1.0
    # Worst item (rank N) -> 0.0
    components = 1.0 - norm_ranks
    
    # Heuristic is the product of components across all constraints
    # This implements the "constraint regret" metric
    heur = np.prod(components, axis=0)
    
    # Handle non-finite values and ensure positivity
    heur = np.where(np.isfinite(heur), heur, 1e-9)
    heur = np.where(heur > 0, heur, 1e-9)
    
    # Time guard
    if time.time() > deadline:
        # Fallback to a simple, fast heuristic if computation is unexpectedly slow
        # This should not happen for N=100, C=5
        sum_weights = weights.sum(axis=0) + eps
        return prizes.sum(axis=0) / sum_weights
        
    return heur
