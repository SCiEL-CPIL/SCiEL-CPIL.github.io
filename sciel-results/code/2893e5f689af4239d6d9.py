"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 300s limit

def _get_elapsed_time():
    return time.time() - _start_time

def _check_deadline():
    return _get_elapsed_time() > (300.0 - _DEADLINE_MARGIN)

def _safe_div(num, den):
    return np.where(den > 1e-9, num / den, 0.0)

def _ei(mu, sigma, f_best):
    z = _safe_div(f_best - mu, sigma)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _ucb(mu, sigma, f_best, kappa):
    # UCB for minimization: we want low mu, high uncertainty helps exploration.
    # Standard UCB for minimization is -mu + kappa*sigma
    # But we need to combine with cost.
    return -mu + kappa * sigma

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid EI/UCB with phase-dependent exploration and softened cost penalty.
    """
    if _check_deadline():
        # Fallback to simple EI/cost if running out of time to ensure valid output quickly
        z = _safe_div(f_best - mu, sigma)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return _safe_div(ei, cost)

    elapsed = _get_elapsed_time()
    
    # Total budget is 30. Initial 5 points cost approx 1 + 0.5 = 1.5 each? 
    # Cost = 1 + mean(x_scaled). Uniform in [0,1]^2 -> mean 0.5. Cost ~ 1.5 per point.
    # 5 points -> cost ~ 7.5.
    # Budget 30. Remaining ~ 22.5.
    # We can estimate the "phase" based on total cost consumed so far.
    # cost array passed is the cost of *evaluating* the candidate? 
    # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # The `cost` input to acquisition is shape (2000,). It likely contains the cost 
    # associated with each of the 2000 candidates if they were evaluated.
    
    # Heuristic for phase:
    # If we assume we've spent ~7.5 already, and limit is 30.
    # Let's use the maximum cost in the batch to gauge where we are? 
    # Actually, we don't know total spent. But we can use a time-based or fixed phase logic.
    # Better: Use the `cost` values to infer relative position? No, cost is property of x.
    
    # Let's use a simple decay based on elapsed time or a fixed schedule.
    # Since we don't have explicit total_cost passed, we rely on the fact that 
    # the loop runs until cost > 30.
    # A robust approach: use a kappa that starts at 1.0 and decays to 0.5.
    
    # Let's estimate progress. 
    # Average cost per step is roughly 1.5.
    # 5 initial steps. 
    # If elapsed time is small, we are early.
    # Time per iteration includes GP fit + 2000 acquisitions.
    # GP fit on N points takes ~ O(N^3). N grows.
    # Early: N=5. Late: N~15.
    # Time is a decent proxy for N.
    
    # Normalize elapsed time to a [0, 1] range assuming ~20-30 iterations max.
    # If we assume ~25 iterations total, and each takes increasing time.
    # Let's just use a fixed schedule based on the number of calls? 
    # We can track calls with a global counter.
    
    global _call_count
    if '_call_count' not in globals():
        _call_count = 0
    _call_count += 1
    
    # Max expected iterations: (30 - 7.5) / 1.5 = 15 iterations.
    # So _call_count goes from 1 to 15.
    # Progress p = _call_count / 15.0
    p = min(_call_count / 15.0, 1.0)
    
    # Kappa decays from 1.0 to 0.2
    kappa = 1.0 - 0.8 * p
    
    # Compute EI
    z = _safe_div(f_best - mu, sigma)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB (for minimization, higher score is better)
    # We want to maximize the score.
    # UCB score: -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma
    
    # Normalize scores to comparable scales?
    # EI is in objective units. UCB is in objective units (mu) + noise units (sigma).
    # They are roughly comparable.
    
    # Blend: weight EI more as we progress?
    # w_ei = 0.5 + 0.5 * p
    # w_ucb = 0.5 - 0.5 * p
    # score = w_ei * ei + w_ucb * ucb_score
    
    # However, cost awareness is crucial.
    # Softened cost penalty: divide by cost^0.5 instead of cost^1.
    # This reduces the bias towards cheap corners (low cost) while still preferring them.
    
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = np.power(cost_safe, 0.5)
    
    # Combine EI and UCB
    # Simple additive blend
    hybrid_score = 0.5 * ei + 0.5 * ucb_score
    
    # Apply cost penalty
    final_score = _safe_div(hybrid_score, cost_penalty)
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score
# EVOLVE-BLOCK-END
