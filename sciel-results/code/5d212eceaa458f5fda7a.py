"""Cost-aware acquisition: Dynamic PI/LCB blend with adaptive cost penalty."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _get_beta():
    """Fixed beta for UCB/LCB exploration."""
    return 2.0

def _get_gamma(total_cost):
    """
    Adaptive cost exponent.
    Early on (low cost), gamma=1.0 (standard 1/cost).
    Late on (high cost), gamma increases to 2.0, heavily penalizing expensive points
    to conserve budget for cheap evaluations.
    """
    # Budget is 30. 
    # Start penalizing more after cost 15.
    # At cost 30, gamma should be ~2.0.
    progress = np.clip(total_cost / 30.0, 0.0, 1.0)
    # Linear ramp from 1.0 to 2.0
    return 1.0 + progress

def _get_weights(total_cost):
    """
    Dynamic weights for PI vs LCB.
    Early: more LCB (exploration/aggressive).
    Late: more PI (exploitation/stability).
    """
    progress = np.clip(total_cost / 30.0, 0.0, 1.0)
    # w_lcb decreases, w_pi increases
    w_lcb = 0.5 * (1.0 - progress) + 0.2  # 0.7 -> 0.2
    w_pi = 1.0 - w_lcb
    return w_pi, w_lcb

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a blend of Probability of Improvement (PI)
    and Lower Confidence Bound (LCB) gain, normalized by an adaptive cost power.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety checks
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # 1. Calculate Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z_pi = (f_best - mu) / sigma_safe
    z_pi = np.clip(z_pi, -10, 10)
    pi = norm.cdf(z_pi)
    
    # 2. Calculate LCB Gain
    # LCB = mu - beta * sigma
    # Gain = max(0, f_best - LCB)
    beta = _get_beta()
    lcb = mu - beta * sigma_safe
    lcb_gain = np.maximum(0.0, f_best - lcb)
    
    # Normalize LCB gain to be in a similar range as PI (0-1)
    # f_best can be large initially, so we normalize by (f_best + 1) or similar
    # Since we are minimizing, f_best starts high.
    # If f_best is close to 0, lcb_gain is small.
    # To make it comparable to PI (0-1), we divide by a scale factor.
    # A simple scale is (f_best + 1.0) to avoid division by zero and keep it bounded roughly.
    scale = np.maximum(f_best, 1.0) + 1.0
    lcb_norm = lcb_gain / scale
    
    # 3. Dynamic Weights
    total_cost = np.mean(cost) # Use mean cost as a proxy for budget used? 
    # Actually, 'cost' is the cost of the *candidate* points. 
    # The prompt says "While total cost is below 30...". 
    # The function receives 'cost' for the 2000 candidates. 
    # It does NOT receive the *current total budget used*.
    # However, we can infer the phase from f_best or just use a static blend?
    # The prompt says "f_best is the lowest observed value".
    # We don't have the total cost spent so far passed to the function.
    # We only have the cost of the *candidates* being evaluated.
    # Therefore, we cannot dynamically adjust based on "time left" unless we track it externally.
    # But the function is stateless per call.
    # We can, however, use f_best as a proxy for progress.
    # If f_best is close to 0, we are late. If f_best is high, we are early.
    # Three-hump camel min is 0.
    # Let's use f_best to determine the "phase".
    # Early: f_best > 1.0. Late: f_best < 0.1.
    
    if f_best > 1.0:
        phase = 0.0
    elif f_best < 0.1:
        phase = 1.0
    else:
        # Linear interpolation between 0.1 and 1.0
        phase = (1.0 - f_best) / (1.0 - 0.1)
        phase = np.clip(phase, 0.0, 1.0)
        
    w_pi, w_lcb = _get_weights(phase)
    
    # 4. Combine Scores
    # PI is [0,1]. LCB_norm is roughly [0,1].
    raw_score = w_pi * pi + w_lcb * lcb_norm
    
    # 5. Cost Normalization
    # Use adaptive gamma based on phase (proxy for budget usage)
    gamma = _get_gamma(phase * 30.0) # Map phase [0,1] to cost [0,30]
    
    # Score = Raw / (Cost ^ Gamma)
    # This penalizes expensive points more when phase is high (late game)
    score = raw_score / (cost_safe ** gamma)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
