"""
Strategy: Adaptive UCB with Cost Normalization and Time-Decaying Exploration.

The current program uses a static blend of EI and LCB. This strategy uses a pure
UCB (Lower Confidence Bound for minimization) with an exploration parameter beta
that decays over time. This allows for aggressive exploration in the early stages
when the GP is uncertain, and shifts to exploitation as the budget is consumed.
The score is normalized by the evaluation cost to favor cheaper evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the progress of the optimization loop
# Each call to 'acquisition' corresponds to one evaluation step in the loop.
_eval_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _eval_call_count
    
    # Increment the global counter
    _eval_call_count += 1
    
    # Estimate the total number of evaluations possible
    # Budget is 30. Min cost is 1 (if x=0). Max cost is 2 (if x=1).
    # Average cost ~1.5. So ~20 evaluations.
    # We cap the decay at 25 steps to be safe.
    max_steps = 25.0
    
    # Calculate the decay factor (0 at start, 1 at end)
    progress = min(_eval_call_count / max_steps, 1.0)
    
    # Define exploration parameters
    # Start with high exploration, end with low exploration (exploitation)
    beta_max = 3.0
    beta_min = 0.5
    
    # Linear decay of beta
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Handle edge cases for sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate UCB score for minimization
    # We want to find low values.
    # UCB_min = mu - beta * sigma
    # To maximize the acquisition score, we maximize (-mu + beta * sigma)
    # This is equivalent to minimizing (mu - beta * sigma)
    ucb_score = -mu + beta * sigma_safe
    
    # Cost awareness
    # Penalize high cost by dividing by cost.
    # Ensure cost is positive and finite
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    scores = ucb_score / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety: If the scores are all identical or invalid, fall back to EI
    # But UCB should be distinct.
    
    return scores
# EVOLVE-BLOCK-END
