# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track wall-clock to avoid being killed.
# We are called repeatedly; each call must be fast.
# Total budget is ~330s wall-clock for the whole evaluation.
# We'll add a safety check but the main work is O(N) with N=2000, so it's fast.

# Global start time for deadline tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: mean of GP predictions, shape (2000,)
    - sigma: std dev of GP predictions, shape (2000,)
    - f_best: best observed function value
    - cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
    - scores: acquisition values, shape (2000,)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero and negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive cost-awareness: strengthen cost penalty when f_best is small (near optimum)
    # This encourages cheaper evaluations in the final phase
    # f_best starts high (around 3-5 for 3-hump camel) and decreases toward 0
    # Use a smooth transition: when f_best < 1.0, increase cost sensitivity
    # cost_factor = 1 + alpha * exp(-f_best / scale) where alpha and scale tune the effect
    # For f_best near 0, cost_factor ~ 1 + alpha
    # For f_best large, cost_factor ~ 1
    
    # Adaptive alpha: stronger cost-awareness near optimum
    f_best_abs = abs(f_best)
    # When f_best is small (close to 0), we want stronger cost penalty
    # Use a sigmoid-like transition
    # transition = 1 / (1 + exp((f_best_abs - threshold) / temp))
    # When f_best_abs < threshold, transition ~ 1 (strong cost penalty)
    # When f_best_abs > threshold, transition ~ 0 (weak cost penalty)
    threshold = 0.5
    temp = 0.2
    transition = 1.0 / (1.0 + np.exp((f_best_abs - threshold) / temp))
    
    # Base cost exponent: 1.0 (standard EI/cost)
    # When near optimum, increase to 1.5 or 2.0 to strongly penalize expensive points
    alpha = 1.0  # max additional exponent
    cost_exp = 1.0 + alpha * transition
    
    # Cost-aware EI with adaptive exponent
    scores = ei / (cost_safe ** cost_exp)
    
    # Exploration bonus in low-EI regions to help escape local minima
    # When EI is very low everywhere, add a small exploration term based on uncertainty
    # This helps when the GP is confident but wrong
    # Exploration bonus: beta * sigma / cost_safe, where beta is small
    # But only when EI is below some threshold relative to f_best
    ei_max = np.max(ei) if len(ei) > 0 else 0.0
    if ei_max > 1e-10:
        # Normalize EI to [0, 1] roughly
        ei_norm = ei / ei_max
        # Exploration bonus is stronger when EI is low
        # Use a soft threshold: bonus = beta * (1 - ei_norm) * sigma / cost
        beta = 0.1
        exploration = beta * (1.0 - np.clip(ei_norm, 0, 1)) * sigma_safe / cost_safe
        scores = scores + exploration
    else:
        # If EI is all ~0, just use pure exploration
        scores = sigma_safe / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
