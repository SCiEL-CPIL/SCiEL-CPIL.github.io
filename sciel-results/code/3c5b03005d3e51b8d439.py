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
    
    # Safety check: if we are running out of time, return a deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    n = len(mu)
    
    # Strategy: Budget-Aware Thompson Sampling with Quadratic Cost Penalty
    #
    # This is a structural change from the current PI/cost^2 approach.
    # Thompson Sampling (TS) samples from the posterior predictive distribution.
    # For minimization, we want to find points with low predicted values.
    # We sample y ~ N(mu, sigma^2).
    # The "value" of a point is -y (higher is better, since we minimize y).
    # To be cost-aware, we normalize by the cost.
    # 
    # Key Insight from Cell Findings: "Switching from EI to LCB provides a structurally 
    # different exploration/exploitation balance... applying the measured quadratic cost 
    # penalty (1/cost^2) significantly increases the number of evaluations within the budget"
    #
    # We will use a Thompson Sampling variant that approximates the expected value of
    # the sample, but with a strong exploration component.
    # Actually, deterministic TS is just the mean. To get exploration, we can use
    # a "Pessimistic" or "Optimistic" bound, or we can use a fixed random seed to
    # sample once? No, acquisition must be deterministic per call (or effectively so).
    #
    # Let's implement **Lower Confidence Bound (LCB)** which is equivalent to the
    # expectation of the minimum of a small number of TS samples, or simply a
    # deterministic approximation of TS that favors exploration.
    # LCB for minimization: mu - beta * sigma.
    # We want to maximize the score, so Score = -(mu - beta * sigma) = -mu + beta * sigma.
    #
    # The cell finding explicitly mentions LCB. Let's use LCB.
    # How to choose beta?
    # A common adaptive beta is based on the number of evaluations or the range.
    # Since we don't have t, we can estimate the "stage" of the search.
    # However, we don't have access to the number of evaluations directly, only f_best and costs.
    # We can estimate the "spread" of the current beliefs.
    #
    # Let's try a fixed beta first, or one that scales with the uncertainty.
    # The finding says "LCB provides a structurally different... balance".
    # Let's try beta = 2.0 or 3.0.
    #
    # Also, the finding emphasizes the **quadratic cost penalty** (1/cost^2).
    # So, Score = (beta * sigma - mu) / cost^2.
    #
    # This is structurally different from PI (which is bounded [0,1]) and EI (which is bounded by range).
    # LCB can be negative if mu is very low, which is good (high score).
    # It encourages exploring high-sigma regions (beta * sigma) and exploiting low-mu regions (-mu).
    
    cost_safe = np.maximum(cost, 1.0)
    
    # LCB: mu - beta * sigma
    # We want to maximize: -LCB / cost^2 = (beta * sigma - mu) / cost^2
    # Let's choose beta = 2.0. This is a common value for BO.
    beta = 2.0
    
    lcb_value = mu - beta * sigma
    # We want to minimize lcb_value.
    # So the score is -lcb_value / cost^2
    score = -lcb_value / (cost_safe ** 2)
    score = beta * sigma / (cost_safe ** 2) - mu / (cost_safe ** 2)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
