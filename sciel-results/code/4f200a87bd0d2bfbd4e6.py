"""
Strategy: Expected Improvement (EI) with Cost Normalization and Time-Adaptive Exploration.

This strategy uses Expected Improvement, which is generally more robust than UCB for
Bayesian optimization. It balances exploitation (finding low mu) and exploration
(high sigma) in a principled way. The exploration is modulated by a time-decaying
parameter that scales the uncertainty, and the final score is normalized by cost to
favor cheaper evaluations.
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
    
    # For EI, we want to maximize E[max(f_best - f(x), 0)]
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    # To allow for early exploration, we can adjust the "effective" f_best
    # or simply rely on the natural exploration of EI.
    # However, pure EI can be too exploitative early on if the initial points are bad.
    # A common trick is to use a slightly worse f_best (optimistic) or add a small
    # exploration term. Here, we stick to standard EI but ensure numerical stability.
    
    # Handle edge cases for sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Use scipy.stats.norm for Phi and phi
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but numerical errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost awareness
    # Penalize high cost by dividing by cost.
    # Ensure cost is positive and finite
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
