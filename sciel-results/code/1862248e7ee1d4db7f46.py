"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: each call should finish well within limits
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: normalized expected improvement
    with adaptive exploration for Ackley 2D minimization.
    
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
    
    if time.monotonic() - t_start > 25:
        # Fallback: simple cost-normalized UCB
        scores = (-mu + 2.0 * sigma) / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (clip tiny negatives from numerical error)
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e4, neginf=0.0)
    
    # Adaptive beta: when f_best is high (far from optimum), explore more
    # Ackley range is roughly [0, ~7], so normalize
    # When f_best is near 0, exploit; when far, explore
    if f_best <= 0:
        beta = 0.5
    elif f_best < 0.1:
        beta = 0.8
    elif f_best < 0.5:
        beta = 1.2
    elif f_best < 2.0:
        beta = 1.8
    elif f_best < 5.0:
        beta = 2.5
    else:
        beta = 3.0
    
    ucb_term = beta * sigma - mu
    ucb_term = np.nan_to_num(ucb_term, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Z-score normalization for robustness (from inspiration 1)
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-10:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_mean = np.mean(ucb_term)
    ucb_std = np.std(ucb_term)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_term - ucb_mean) / ucb_std
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Blend: weight shifts from exploration to exploitation as f_best improves
    # Use a smooth transition
    # f_best = 0 -> w_ei = 0.9, w_ucb = 0.1
    # f_best = 10 -> w_ei = 0.3, w_ucb = 0.7
    if f_best <= 0:
        w_ei = 0.9
        w_ucb = 0.1
    else:
        # Smooth decay of exploration weight
        w_ucb = 0.1 + 0.6 * (1.0 - np.exp(-f_best / 2.0))
        w_ei = 1.0 - w_ucb
    
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware: divide by cost. 
    # Cost ranges from 1 (at x=0) to 2 (at x=5) in each dim, so 1 to ~2.
    # Dividing by cost penalizes expensive evaluations.
    scores = combined / cost
    
    # Additional shaping: boost candidates that are far from the current best
    # in terms of expected minimum value. This helps escape local optima.
    # The GP lower bound is mu - 2*sigma (approx). Prefer where this is very low.
    # But this is already captured in UCB. Let's add a small boost for high sigma
    # candidates to ensure exploration.
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.monotonic() - t_start > 28:
        return scores
    
    # Optional: apply a mild sharpening to make the best candidates stand out more
    # This helps the "pick best of 2000" strategy
    # Sort and apply a power transformation to the top scores
    # But be careful: we need to return scores for ALL 2000 candidates
    
    # A simple approach: amplify the top few percent
    # Use a threshold: scores above median get a boost
    med = np.median(scores)
    if med > 0:
        # Apply a gentle nonlinearity: score^alpha where alpha > 1 amplifies top
        # But only for positive scores
        pos_mask = scores > 0
        if np.any(pos_mask):
            # Normalize positive scores to [0, 1] then apply power
            max_pos = np.max(scores[pos_mask])
            if max_pos > 1e-10:
                scores[pos_mask] = (scores[pos_mask] / max_pos) ** 1.3 * max_pos
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
