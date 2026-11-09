"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Ackley 2D minimization.
    
    Combines:
    - Probability of Improvement (PI) for smooth exploration gradients
    - Expected Improvement (EI) for exploitation signal
    - Adaptive UCB for exploration in early stages
    
    Uses Z-score normalization for stability and smooth adaptive weighting
    that shifts from exploration to exploitation as f_best approaches 0.
    Normalized by sqrt(cost) to balance exploration efficiency.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (float)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    t_start = time.monotonic()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Deadline guard
    if time.monotonic() - t_start > 20:
        scores = (-mu + 2.0 * sigma) / np.sqrt(cost)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Probability of Improvement for minimization
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    improvement = f_best - mu
    z = improvement / sigma
    pi = norm.cdf(z)
    pi = np.clip(pi, 0.0, 1.0)
    pi = np.nan_to_num(pi, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Expected Improvement for exploitation signal
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e4, neginf=0.0)
    
    # Adaptive beta based on f_best - more exploration when far from optimum
    # Smooth transition: beta decreases as f_best approaches 0
    if f_best <= 0:
        beta = 0.3
    else:
        beta = 0.3 + 2.2 * (1.0 - np.exp(-f_best / 2.0))
    
    ucb_term = beta * sigma - mu
    ucb_term = np.nan_to_num(ucb_term, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Z-score normalization for robustness (from inspirations)
    # This provides stable gradients regardless of scale
    
    # PI is already in [0, 1], but normalize for consistent blending
    pi_mean = np.mean(pi)
    pi_std = np.std(pi)
    if pi_std > 1e-10:
        pi_norm = (pi - pi_mean) / pi_std
    else:
        pi_norm = np.zeros_like(pi)
    pi_norm = np.nan_to_num(pi_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # EI normalization
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    ei_norm = np.nan_to_num(ei_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # UCB normalization
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_term - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb_term)
    ucb_norm = np.nan_to_num(ucb_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Adaptive weights with smooth transition (from inspiration 2)
    # Early (high f_best): explore more with UCB and PI
    # Late (low f_best): exploit with EI
    if f_best <= 0:
        w_pi = 0.4
        w_ei = 0.5
        w_ucb = 0.1
    else:
        # Smooth decay: as f_best increases, shift weight toward exploration
        exploration_factor = 1.0 - np.exp(-f_best / 3.0)
        w_ucb = 0.1 + 0.4 * exploration_factor
        w_pi = 0.2 + 0.3 * exploration_factor
        w_ei = 1.0 - w_ucb - w_pi
    
    # Ensure weights are valid
    w_pi = max(w_pi, 0.05)
    w_ei = max(w_ei, 0.1)
    w_ucb = max(w_ucb, 0.05)
    w_sum = w_pi + w_ei + w_ucb
    w_pi /= w_sum
    w_ei /= w_sum
    w_ucb /= w_sum
    
    combined = w_pi * pi_norm + w_ei * ei_norm + w_ucb * ucb_norm
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Cost-aware scoring: divide by sqrt(cost)
    # Cost ranges from 1 to 2, so sqrt(cost) ranges from 1 to ~1.41
    scores = combined / np.sqrt(cost)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.monotonic() - t_start > 28:
        return scores
    
    # Mild sharpening to emphasize top candidates
    # Apply power transformation to positive scores to make best stand out
    pos_mask = scores > 0
    if np.any(pos_mask):
        max_pos = np.max(scores[pos_mask])
        if max_pos > 1e-10:
            # Normalize to [0, 1] then apply power > 1 to amplify top
            normalized = scores[pos_mask] / max_pos
            scores[pos_mask] = normalized ** 1.3 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
