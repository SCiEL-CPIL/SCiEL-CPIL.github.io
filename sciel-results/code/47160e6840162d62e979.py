"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds to leave before the 330s limit

def _check_deadline():
    return (time.time() - _start_time) > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Multi-Scale Cost-Aware Expected Improvement.
    
    This strategy uses an ensemble of EI calculations with different 
    exploration-exploitation balances (via a scaling factor on sigma) 
    to capture both high-confidence improvements and promising high-uncertainty 
    regions. It is divided by cost to account for the variable evaluation cost.
    """
    if _check_deadline():
        # Fallback to simple EI if we are running out of time
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensemble of EI with different exploration levels
    # We scale sigma by different factors to simulate different risk appetites.
    # Smaller sigma scale -> more exploitation (trusts mu more)
    # Larger sigma scale -> more exploration (trusts uncertainty more)
    
    # 1. Pure EI (standard)
    z1 = (f_best - mu) / sigma_safe
    ei1 = (f_best - mu) * norm.cdf(z1) + sigma_safe * norm.pdf(z1)
    
    # 2. Conservative EI (exploit high confidence)
    # Effectively reduces the impact of uncertainty, favoring points where mu is already good
    sigma_cons = sigma_safe * 0.5
    z2 = (f_best - mu) / (sigma_cons + 1e-9)
    ei2 = (f_best - mu) * norm.cdf(z2) + sigma_cons * norm.pdf(z2)
    
    # 3. Aggressive EI (explore high uncertainty)
    # Effectively increases the impact of uncertainty, favoring points with high sigma
    sigma_agg = sigma_safe * 2.0
    z3 = (f_best - mu) / (sigma_agg + 1e-9)
    ei3 = (f_best - mu) * norm.cdf(z3) + sigma_agg * norm.pdf(z3)
    
    # Combine the scores. 
    # We weight exploitation more heavily because we have a limited budget (30 cost units).
    # However, some exploration is necessary to escape local minima in complex landscapes like Ackley or Rastrigin.
    
    # Weighting: 0.6 for standard, 0.3 for conservative, 0.1 for aggressive
    # This creates a bias towards reliable improvements while still allowing jumps to uncertain areas.
    
    combined_ei = 0.6 * ei1 + 0.3 * ei2 + 0.1 * ei3
    
    # Normalize by cost
    scores = combined_ei / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
