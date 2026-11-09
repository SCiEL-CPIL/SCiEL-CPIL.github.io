"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0

def _check_deadline():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_DEADLINE - 1.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier filtering and hypervolume contribution.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    if _check_deadline():
        return np.zeros(len(mu))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standard normal variable
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Handle invalid costs
    cost_safe = np.where(cost > 0, cost, 1.0)
    
    # Structural Change: Pareto Frontier Filter + Hypervolume Contribution
    # Objectives: Maximize EI, Minimize Cost
    
    # 1. Identify Pareto-optimal candidates
    # A candidate i is dominated if there exists j such that EI_j >= EI_i and Cost_j <= Cost_i (and at least one strict)
    
    # Sort by EI descending
    order = np.argsort(-ei)
    ei_sorted = ei[order]
    cost_sorted = cost_safe[order]
    
    # Find min cost in the entire set to use as reference for hypervolume
    max_cost_ref = np.max(cost_safe)
    
    # Initialize dominated mask
    dominated = np.zeros(len(ei), dtype=bool)
    
    # Efficient dominance check using sorting
    # Iterate through sorted EI (descending). 
    # Keep track of the minimum cost seen so far (among higher EI candidates).
    # If current cost is strictly greater than the min cost seen so far, it is dominated.
    min_cost_so_far = np.inf
    # We need to handle ties in EI carefully.
    # If EI is equal, the one with lower cost is better.
    
    # To be safe and simple with N=2000, we can do a vectorized check or a simple loop.
    # N=2000 is small enough for a simple loop in Python if optimized, or vectorized logic.
    
    # Vectorized approach for dominance:
    # i is dominated if max(EI[j] for j where Cost[j] <= Cost[i]) > EI[i] 
    # OR if any j has EI[j] >= EI[i] and Cost[j] < Cost[i]
    
    # Let's use the sorting method which is O(N log N)
    # We mark a point as dominated if its cost is higher than the minimum cost of any point with higher or equal EI.
    
    # Re-initialize
    dominated_sorted = np.zeros(len(ei_sorted), dtype=bool)
    
    # We traverse from highest EI to lowest.
    # We maintain a running minimum of costs.
    # However, if multiple points have the same EI, the one with the lowest cost is the "best" and should not be dominated by others with same EI but higher cost.
    
    # Let's group by EI? No, just sort by EI desc, then Cost asc as secondary key.
    # If we sort by EI desc, Cost asc:
    # The first point in a group of equal EI is the best for that EI level.
    # Any subsequent point with same EI is dominated by the first one (since cost is higher or equal).
    # A point with lower EI is dominated if its cost is higher than the min cost seen so far in higher EI groups.
    
    # Sort indices: primary EI desc, secondary Cost asc
    # np.lexsort takes keys in reverse order of priority
    sort_keys = (cost_safe, -ei) 
    # Wait, lexsort: last key is primary.
    # We want EI primary (desc), Cost secondary (asc).
    # So keys should be (cost_safe, -ei) -> last key -ei is primary?
    # No, np.lexsort((a, b)) sorts by b first, then a.
    # So we want to sort by -ei first? No, we want EI desc.
    # Let's just use np.argsort with a combined key or stable sort.
    
    # Simpler: 
    # 1. Sort by EI descending.
    # 2. Within same EI, sort by Cost ascending.
    # This ensures that for any EI level, the cheapest point comes first.
    
    # Using np.lexsort:
    # Keys: (cost, -ei)
    # Sorts by -ei (primary, so EI desc), then cost (secondary, so Cost asc).
    order = np.lexsort((cost_safe, -ei))
    
    ei_sorted = ei[order]
    cost_sorted = cost_safe[order]
    
    # Now, iterate to find dominated points
    # A point is dominated if there is a previous point (in this sorted order) that has:
    # 1. Higher EI (strictly, because if EI is same, the previous one has lower/equal cost, so current is dominated if cost is higher)
    # 2. Lower Cost (strictly, because if cost is same, previous has higher/equal EI)
    
    # Actually, in this sorted order (EI desc, Cost asc):
    # Any point that is not the "best" in its EI group is dominated by the first in that group.
    # Any point with lower EI is dominated if its cost is >= min_cost_so_far.
    
    min_cost_so_far = np.inf
    dominated_sorted = np.zeros(len(ei_sorted), dtype=bool)
    
    for i in range(len(ei_sorted)):
        if cost_sorted[i] >= min_cost_so_far:
            # This point is dominated by some point with higher (or equal) EI and lower (or equal) cost
            # Since we sorted by EI desc, Cost asc:
            # If EI is same, cost is higher than the first one -> Dominated.
            # If EI is lower, cost is higher than min cost of higher EI -> Dominated.
            dominated_sorted[i] = True
        else:
            min_cost_so_far = cost_sorted[i]
            
    # Map back to original indices
    dominated = np.zeros(len(ei), dtype=bool)
    dominated[order] = dominated_sorted
    
    # 2. Compute Hypervolume Contribution for non-dominated points
    # Reference point: I_ref = 0, C_ref = max(cost)
    # Contribution = EI * (C_ref - Cost)
    
    hv_contrib = ei * (max_cost_ref - cost_safe)
    
    # Set scores
    scores = np.where(dominated, 0.0, hv_contrib)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
