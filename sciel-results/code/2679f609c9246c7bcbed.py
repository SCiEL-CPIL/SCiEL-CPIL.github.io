import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 320.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, f_best, beta):
    """Compute UCB score for minimization: -mu + beta * sigma."""
    return -mu + beta * sigma


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lookahead-UCB: a two-stage hierarchical score.
    
    Stage 1 (coarse): UCB-based ranking to identify the top-k most promising
    candidates, where k shrinks over time (more exploration early, less later).
    Stage 2 (refined): Within the top-k, use EI to pick the best exploitation
    target. Candidates outside top-k get a small residual score.
    
    This is structurally different from a linear blend: the top-k gate creates
    a hard threshold that changes the effective search space at each step.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # Adaptive UCB beta: start high (exploration), decay to low (exploitation)
    # beta from ~3.0 early to ~0.5 late
    beta = 3.0 - 2.5 * progress
    
    # Stage 1: UCB scores for coarse ranking
    ucb = _ucb_score(mu, sigma_safe, f_best, beta)
    
    # Adaptive top-k fraction: 60% of candidates early, 15% late
    k_frac = 0.6 - 0.45 * progress
    k = max(50, int(k_frac * len(mu)))
    
    # Find indices of top-k UCB candidates
    top_k_idx = np.argpartition(ucb, -k)[-k:]
    
    # Stage 2: EI scores for refined selection within top-k
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Build final score: high EI within top-k, low residual outside
    # Normalize EI within the top-k subset to [0, 1]
    ei_topk = ei[top_k_idx]
    ei_min = np.min(ei_topk)
    ei_max = np.max(ei_topk)
    ei_range = ei_max - ei_min
    if ei_range < _EPS:
        ei_norm_topk = np.ones_like(ei_topk)
    else:
        ei_norm_topk = (ei_topk - ei_min) / ei_range
    
    # Final score array
    score = np.zeros(len(mu))
    
    # Within top-k: score = 1.0 + normalized EI (range [1, 2])
    score[top_k_idx] = 1.0 + ei_norm_topk
    
    # Outside top-k: small residual based on normalized UCB (range [0, 0.5))
    # This ensures the argmax is always within top-k, but preserves some
    # gradient information for the optimizer
    outside_mask = np.ones(len(mu), dtype=bool)
    outside_mask[top_k_idx] = False
    if np.any(outside_mask):
        ucb_out = ucb[outside_mask]
        ucb_min_out = np.min(ucb_out)
        ucb_max_out = np.max(ucb_out)
        ucb_range_out = ucb_max_out - ucb_min_out
        if ucb_range_out < _EPS:
            score[outside_mask] = 0.25
        else:
            score[outside_mask] = 0.5 * (ucb_out - ucb_min_out) / ucb_range_out
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
