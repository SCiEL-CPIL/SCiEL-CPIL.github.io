"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a simple deterministic EI
    if _time_exceeded():
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Monte Carlo Expected Improvement (MC-EI) with Adaptive Cost Penalty
    #
    # Rationale:
    # 1. Standard EI assumes an infinite horizon or ignores the cost of the next evaluation.
    # 2. This task has a strict cost budget (30 units).
    # 3. We approximate the Expected Improvement by sampling from the GP posterior.
    #    This is a "structural" change from closed-form EI because it allows us to
    #    more easily incorporate complex constraints or non-linear cost interactions
    #    if needed, and it is robust to numerical issues in the tail of the normal
    #    distribution that can sometimes plague closed-form implementations.
    # 4. We introduce an "Adaptive Cost Penalty" that scales the acquisition score
    #    by the ratio of remaining budget to current cost. As the budget runs low,
    #    the penalty for high-cost evaluations increases, forcing the optimizer
    #    to choose cheaper points (or points with much higher expected improvement).
    #
    # The "cost" is defined as 1 + mean(x scaled to [0,1]).
    # We don't have x, but we have the pre-computed 'cost' array.
    
    # Estimate the "urgency" of the budget.
    # We don't know the exact total cost spent so far, but we can infer the stage
    # from the cost of the candidates and f_best.
    # A simpler, robust heuristic:
    # If f_best is close to the known minimum (0 or -3.86), we should be very conservative.
    # However, we don't know the function type.
    #
    # Let's use a fixed but principled approach:
    # Score = (Expected Improvement) / (cost^alpha)
    # Where alpha is chosen to balance exploration vs cost.
    # In the previous attempt, alpha=2 (quadratic) was used with LCB.
    # Here, we use EI (which is more exploitative than LCB) but with a softer penalty
    # initially, increasing as we assume we are deeper in the search.
    
    # Since we cannot track the exact number of evaluations, we use the magnitude of f_best
    # and the spread of sigma to estimate progress.
    # If sigma is small everywhere, we are in late stages -> higher alpha.
    # If sigma is large, we are in early stages -> lower alpha.
    
    sigma_mean = np.mean(sigma)
    # Normalize sigma to [0, 1] roughly based on typical GP behavior.
    # If sigma_mean is very small, we are converged.
    progress_estimate = 1.0 / (1.0 + sigma_mean * 10.0)
    
    # Alpha ranges from 1.0 (early, cost matters less) to 2.5 (late, cost matters a lot)
    alpha = 1.0 + 1.5 * progress_estimate
    
    # Calculate Expected Improvement using Monte Carlo for robustness and flexibility
    # We use a small number of samples (e.g., 10) to approximate the integral.
    # This is faster than 1000 samples and more robust than closed-form if sigma varies wildly.
    # Actually, closed-form EI is O(N) and very fast. MC is O(N * S).
    # Let's stick to closed-form EI for speed, but modify the *interpretation* of the score.
    
    # The key structural change: **Probability of Improvement (PI) blended with EI**
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # EI = (f_best - mu) * PI + sigma * phi((f_best - mu) / sigma)
    #
    # The current program used LCB. LCB is good for exploration.
    # EI is good for exploitation.
    # PI is purely probabilistic.
    #
    # Let's try a **Cost-Aware PI-EI Hybrid**:
    # Score = (w * EI + (1-w) * PI * sigma) / cost^alpha
    # Where w adapts to the uncertainty.
    # High uncertainty -> higher weight on PI (exploration).
    # Low uncertainty -> higher weight on EI (exploitation).
    
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Weight for exploration (PI) vs exploitation (EI)
    # If sigma is high, we want to explore (PI).
    # If sigma is low, we want to exploit (EI).
    # Normalize sigma to get a weight.
    # Use a sigmoid-like transition.
    sigma_norm = sigma / (np.max(sigma) + 1e-9)
    w_exploration = 0.5 + 0.5 * sigma_norm  # 0.5 to 1.0
    w_exploitation = 1.0 - w_exploration    # 0.0 to 0.5
    
    # The "Value" of the point
    # PI is in [0,1]. EI is in [0, infinity).
    # To make them comparable, we scale PI by the typical improvement scale.
    # A reasonable scale is the current sigma or the range of mu.
    scale = np.mean(sigma) + 1e-9
    value = w_exploitation * ei + w_exploration * (pi * scale)
    
    # Apply cost penalty
    cost_safe = np.maximum(cost, 1.0)
    score = value / (cost_safe ** alpha)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
