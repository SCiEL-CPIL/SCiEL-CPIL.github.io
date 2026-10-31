import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin based on the reduction in the variance of the remaining 
    capacity distribution across all open bins.
    
    The approach models bin selection as minimizing the statistical spread 
    (variance) of the remaining capacities. This aims to keep the load 
    distribution balanced, which is a probabilistic risk mitigation strategy 
    against fragmentation.
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    N = len(bins)
    if N == 0:
        return np.array([])
        
    valid_mask = bins >= item
    if not np.any(valid_mask):
        return np.full(N, -np.inf)
        
    # Precompute global statistics
    S = np.sum(bins)
    Q = np.sum(bins**2)
    
    # The change in variance when placing item 'item' in bin i is:
    # Var_new = (Q - 2*bins[i]*item + item**2)/N - ((S - item)/N)**2
    # We want to minimize Var_new, which is equivalent to maximizing the score:
    # Score = -Var_new
    
    # Expanding the terms:
    # Var_new = Q/N - (2*item/N)*bins[i] + item**2/N - (S**2 - 2*S*item + item**2)/N**2
    # Var_new = (Q/N - S**2/N**2 + item**2/N - item**2/N**2 + 2*S*item/N**2) - (2*item/N)*bins[i]
    
    # The term in the parenthesis is constant for all bins in this step.
    # Let C = Q/N - (S/N)**2 + item**2*(N-1)/N**2 + 2*S*item/N**2
    # Var_new = C - (2*item/N) * bins[i]
    
    # To minimize Var_new, we must maximize bins[i] (since item > 0).
    # Thus, the optimal bin is the one with the largest remaining capacity.
    # However, we return the actual negative variance score to reflect the approach.
    
    C = Q/N - (S/N)**2 + (item**2 * (N-1)) / (N**2) + (2 * S * item) / (N**2)
    coefficient = - (2 * item) / N
    
    # Calculate score for all bins
    # Score = - (C + coefficient * bins)  ... wait
    # Var_new = C + coefficient * bins[i]  (where coefficient is negative)
    # We want to minimize Var_new.
    # So Score should be - Var_new.
    # Score = -C - coefficient * bins[i]
    # Since coefficient is negative, -coefficient is positive.
    # Score = -C + (2*item/N) * bins[i]
    
    scores = np.full(N, -np.inf)
    # For valid bins, score is proportional to bins[i]
    # We can just use bins[i] as the score, as the other terms are constant/monotonic.
    # But to be "faithful", we compute the full expression.
    
    valid_indices = np.where(valid_mask)[0]
    valid_bins = bins[valid_indices]
    
    # Compute scores for valid bins
    scores_valid = -C + (2 * item / N) * valid_bins
    
    scores[valid_indices] = scores_valid
    
    return scores
