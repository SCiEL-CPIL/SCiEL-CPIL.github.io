"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a smooth combination of Expected Improvement (EI) and multi-scale UCB
    to balance exploitation and exploration for the multi-modal Branin function.
    The soft-max over UCB scales provides a smooth surrogate that avoids the
    discontinuities of hard max, while the weighted blend with EI maintains
    strong exploitation pressure.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute EI normalization
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Multi-scale UCB: try several beta values and combine smoothly
    # Using a soft-max (log-sum-exp) over normalized UCB scores provides
    # a smooth surrogate for the hard max, which can be nonsmooth
    betas = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
    
    # Compute UCB scores for each beta: -mu + beta * sigma (higher is better for minimization)
    ucb_scores = -mu[None, :] + betas[:, None] * sigma_safe[None, :]
    
    # Normalize each beta's UCB scores to [0, 1]
    ucb_scores_norm = np.zeros_like(ucb_scores)
    for i in range(len(betas)):
        row = ucb_scores[i, :]
        row_max = np.max(row)
        row_min = np.min(row)
        if row_max - row_min > 1e-12:
            ucb_scores_norm[i, :] = (row - row_min) / (row_max - row_min)
    
    # Soft-max over beta dimension: smooth approximation to max
    # This avoids the nonsmoothness of np.max and provides a differentiable
    # combination that emphasizes the best-performing beta while maintaining
    # sensitivity to all scales
    # Temperature parameter controls how "sharp" the soft-max is
    temperature = 0.1
    # Shift for numerical stability
    shifted = ucb_scores_norm / temperature
    shifted_max = np.max(shifted, axis=0, keepdims=True)
    exp_shifted = np.exp(shifted - shifted_max)
    soft_max = temperature * (np.log(np.sum(exp_shifted, axis=0)) + shifted_max[0])
    
    # The soft-max is in the same scale as the normalized UCB scores [0,1]
    # Scale it back to EI's scale for meaningful combination
    if ei_max > 1e-12:
        ucb_scaled = soft_max * ei_max
    else:
        ucb_scaled = np.zeros_like(soft_max)
    
    # Final score: weighted combination of EI and smooth multi-scale UCB
    # Weight 0.75 on EI and 0.25 on UCB to favor exploitation while
    # maintaining exploration for multi-modal functions
    score = 0.75 * ei + 0.25 * ucb_scaled
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
