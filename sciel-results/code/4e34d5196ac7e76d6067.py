"""
Strategy: Cost-Adjusted Expected Improvement with Pure Exploitation Bias.

This strategy diverges from the current EI/LCB blend by focusing on a
cost-adjusted Expected Improvement (EI) with a strong exploitation bias.
The key insight is that for Powell 4D, which is smooth and unimodal-ish,
greedy exploitation of the GP's mean (mu) is often superior to heavy
exploration (LCB) in the later stages. The cost penalty is applied as a
simple division by cost, which is standard and effective.

We use a pure EI score divided by cost. EI naturally balances exploration
and exploitation: it is high when sigma is large (exploration) and when
mu is low (exploitation). By dividing by cost, we prefer cheaper points
that have high expected improvement.

We do not use LCB to avoid over-exploration in a smooth function.
We do not use adaptive beta because EI already adapts via sigma.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the number of acquisition calls
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    start_time = time.time()
    _call_count += 1
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2), EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-adjusted EI
    # We divide by cost to prefer cheaper evaluations with high EI.
    # This is the standard "cost-aware EI" or "cEI".
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
         # Fallback: just return EI without cost adjustment if something went wrong
         return ei
    
    return scores
# EVOLVE-BLOCK-END
