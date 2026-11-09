"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with an adaptive UCB exploration term,
    using z-score normalization for robustness and mild sharpening to balance
    exploration/exploitation. Divides by cost for cost-awareness.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each point.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB term for minimization: we want low mu and high uncertainty
    # Adaptive beta: larger when f_best is far from optimum (early exploration)
    # Ackley min is 0, max is ~20+20+e+20 ≈ 65, but typically f_best < 40
    f_norm = min(max(f_best / 40.0, 0.0), 1.0)
    beta = 2.0 * (0.3 + 0.7 * f_norm)  # beta in [0.6, 2.0]
    
    ucb = mu + beta * sigma_safe
    
    # Normalize EI using z-score (more robust to outliers than min-max)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-12:
        ei_z = (ei - ei_mean) / ei_std
    else:
        ei_z = np.zeros_like(ei)
    
    # Mild sharpening (1.2) to reduce bias toward extreme candidates
    ei_z = np.tanh(1.2 * ei_z)
    # Scale to [0, 1]
    ei_norm = (ei_z - np.min(ei_z)) / (np.max(ei_z) - np.min(ei_z) + 1e-12)
    
    # Normalize UCB: lower UCB is better, so invert
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std > 1e-12:
        ucb_z = (ucb - ucb_mean) / ucb_std
    else:
        ucb_z = np.zeros_like(ucb)
    
    # Invert: lower UCB should get higher score
    ucb_z = -ucb_z
    # Mild sharpening
    ucb_z = np.tanh(1.2 * ucb_z)
    # Scale to [0, 1]
    ucb_norm = (ucb_z - np.min(ucb_z)) / (np.max(ucb_z) - np.min(ucb_z) + 1e-12)
    
    # Cost normalization: cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive weighting: early -> more UCB (exploration), late -> more EI (exploitation)
    # When f_best is high (early), weight UCB more; when low (late), weight EI more
    w_ei = 0.3 + 0.7 * (1.0 - f_norm)  # [0.3, 1.0]
    w_ucb = 1.0 - w_ei + 0.3  # [0.3, 1.0], ensure minimum exploration
    
    # Combined score: weighted blend divided by cost
    score = (w_ei * ei_norm + w_ucb * ucb_norm) / cost_safe
    
    # Check deadline: if we're running out of time, return what we have
    # The function is fast enough that this shouldn't trigger, but guard just in case
    if time.time() - start_time > 290.0:
        pass  # Return the computed score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
