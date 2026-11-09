"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier filtering
    and Hypervolume contribution scoring.
    
    Args:
        mu (np.ndarray): Mean of GP posterior, shape (N,).
        sigma (np.ndarray): Std dev of GP posterior, shape (N,).
        f_best (float): Best observed function value.
        cost (np.ndarray): Cost of evaluating each candidate, shape (N,).
        
    Returns:
        np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin for 300s limit
    
    # 1. Compute Expected Improvement (EI)
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is only useful if it's positive
    ei = np.maximum(ei, 0.0)
    
    # 2. Sort candidates by Cost (ascending)
    # We want to maximize EI and minimize Cost
    sort_indices = np.argsort(cost)
    sorted_cost = cost[sort_indices]
    sorted_ei = ei[sort_indices]
    
    # 3. Identify Pareto Frontier
    # A point is on the Pareto frontier if its EI is strictly greater than
    # the max EI of all points with lower or equal cost.
    # Since we sorted by cost ascending, we can use cummax.
    max_ei_so_far = np.maximum.accumulate(sorted_ei)
    
    # Points where EI == max_ei_so_far are the "record breakers" in EI
    # as cost increases. These form the Pareto front.
    # Note: If EI is equal to the previous max, it is NOT strictly better,
    # so it is dominated (same EI, higher cost). 
    # np.maximum.accumulate keeps the first occurrence of the max.
    # So sorted_ei[i] == max_ei_so_far[i] implies it's the first time this EI value appeared.
    is_pareto = (sorted_ei == max_ei_so_far)
    
    # Check deadline
    if time.time() - start_time > deadline:
        return np.zeros_like(ei)
        
    # 4. Calculate Hypervolume Contribution for Pareto Points
    C_max = 2.0  # Maximum possible cost
    
    # Extract Pareto points
    pareto_indices_in_sorted = np.where(is_pareto)[0]
    
    if len(pareto_indices_in_sorted) == 0:
        return np.zeros_like(ei)
        
    pareto_eis = sorted_ei[pareto_indices_in_sorted]
    pareto_costs = sorted_cost[pareto_indices_in_sorted]
    
    # Calculate "heights" (difference in EI from previous Pareto point)
    # The "previous" EI for the first point is 0 (reference EI)
    prev_pareto_eis = np.zeros_like(pareto_eis)
    if len(pareto_eis) > 1:
        prev_pareto_eis[1:] = pareto_eis[:-1]
        
    heights = pareto_eis - prev_pareto_eis
    
    # Calculate "widths" (cost range covered by this point's contribution)
    # The width is the distance from this point's cost to the max cost.
    # This represents the area of the rectangle [prev_EI, EI] x [Cost, C_max]
    widths = C_max - pareto_costs
    
    # HV Contribution Score
    pareto_scores = heights * widths
    
    # 5. Map scores back to original indices
    scores = np.zeros_like(ei)
    original_pareto_indices = sort_indices[pareto_indices_in_sorted]
    scores[original_pareto_indices] = pareto_scores
    
    return scores
# EVOLVE-BLOCK-END
