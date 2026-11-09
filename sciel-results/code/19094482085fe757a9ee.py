# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_N_CAND = 2000

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    Uses a combination of Expected Improvement and Probability of Improvement
    with cost normalization and adaptive exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    start_time = time.time()
    deadline = start_time + 295.0
    
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Safety: ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Cost-aware PI
    score_pi = pi / cost_safe
    
    # LCB-style: lower mean + higher uncertainty is better
    # Use a beta that adapts: high when f_best is large (early), low when f_best is small
    # beta = 2.5 * (1 + f_best) / (1 + f_best/5) -- smooth transition
    beta = 2.0 * (1.0 + f_best) / (1.0 + f_best / 3.0)
    lcb = mu - beta * sigma_safe
    score_lcb = -lcb / cost_safe  # Negate so higher = better
    
    # Normalize each component to [0, 1] range for balanced combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        rng = mx - mn
        if rng < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / rng
    
    ei_norm = normalize(score_ei)
    pi_norm = normalize(score_pi)
    lcb_norm = normalize(score_lcb)
    
    # Weights: 
    # - EI is the primary exploitation signal
    # - PI provides a robust improvement probability
    # - LCB encourages exploration in uncertain regions
    # 
    # As f_best approaches 0, we should focus more on exploitation (EI)
    # When f_best is large, more exploration (LCB, PI) is beneficial
    
    # f_best for Levy 2D starts around 10-20 and goes to 0
    # Use a smooth transition based on f_best
    f_best_clamped = max(f_best, 0.0)
    
    # Transition parameter: 0 when f_best is large, 1 when f_best is near 0
    t = 1.0 / (1.0 + f_best_clamped / 2.0)
    
    # Weights that shift from exploration to exploitation
    w_ei = 0.4 + 0.4 * t      # 0.4 to 0.8
    w_pi = 0.3 * (1 - t) + 0.1  # 0.3 to 0.1
    w_lcb = 0.3 * (1 - t) + 0.1  # 0.3 to 0.1
    
    # Normalize weights
    w_sum = w_ei + w_pi + w_lcb
    w_ei /= w_sum
    w_pi /= w_sum
    w_lcb /= w_sum
    
    scores = w_ei * ei_norm + w_pi * pi_norm + w_lcb * lcb_norm
    
    # Add a small bonus for high uncertainty to avoid premature convergence
    max_sigma = np.max(sigma_safe)
    if max_sigma > 1e-12:
        uncertainty_bonus = 0.05 * (sigma_safe / max_sigma) * (1.0 - t)
    else:
        uncertainty_bonus = 0.0
    
    scores = scores + uncertainty_bonus
    
    # Time check
    if time.time() > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
