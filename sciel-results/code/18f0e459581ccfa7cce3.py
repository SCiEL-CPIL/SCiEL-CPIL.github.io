"""Bayesian optimization acquisition for 2D RBF mixture with Matérn-5/2 GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 310.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


def _kriging_believer_score(mu, sigma, progress):
    """
    Kriging-Believer quantile score for minimization.
    Returns -q_tau where q_tau is the tau-quantile of N(mu, sigma^2).
    Low tau -> very negative quantile -> high score (explore).
    High tau (->0.5) -> mean -> exploit.
    """
    sigma_safe = np.maximum(sigma, _EPS)
    # tau goes from 0.05 (early, explore) to 0.5 (late, exploit)
    tau = 0.05 + 0.45 * (1.0 - np.exp(-3.0 * progress))
    tau = np.clip(tau, 0.01, 0.5)
    z_tau = norm.ppf(tau)
    q_tau = mu + sigma_safe * z_tau
    return -q_tau


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Two-stage hierarchical acquisition:
    Stage 1 (coarse gate): Kriging-Believer quantile ranking to select top-k
    candidates. k shrinks over time (60% early -> 15% late).
    Stage 2 (refined): Within top-k, use EI for fine-grained exploitation.
    Outside top-k, a small residual KB score preserves gradient info.
    
    This structural two-stage approach (from Lookahead-UCB) combined with the
    Kriging-Believer quantile signal (from current program) creates a hard
    exploration gate that prevents the optimizer from being pulled by
    high-EI-but-low-potential points in later iterations.
    
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
    
    # Stage 1: Kriging-Believer score for coarse gating
    kb_score = _kriging_believer_score(mu, sigma_safe, progress)
    
    # Adaptive top-k: 60% of candidates early, 15% late
    k_frac = 0.6 - 0.45 * progress
    k = max(50, int(k_frac * len(mu)))
    k = min(k, len(mu))
    
    # Find indices of top-k KB candidates
    top_k_idx = np.argpartition(kb_score, -k)[-k:]
    
    # Stage 2: EI for refined selection within top-k
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Build final score
    score = np.zeros(len(mu))
    
    # Within top-k: score = 1.0 + normalized EI (range [1, 2])
    ei_topk = ei[top_k_idx]
    ei_min = np.min(ei_topk)
    ei_max = np.max(ei_topk)
    ei_range = ei_max - ei_min
    if ei_range < _EPS:
        ei_norm_topk = np.ones_like(ei_topk)
    else:
        ei_norm_topk = (ei_topk - ei_min) / ei_range
    
    score[top_k_idx] = 1.0 + ei_norm_topk
    
    # Outside top-k: small residual based on normalized KB score (range [0, 0.5))
    outside_mask = np.ones(len(mu), dtype=bool)
    outside_mask[top_k_idx] = False
    if np.any(outside_mask):
        kb_out = kb_score[outside_mask]
        kb_min_out = np.min(kb_out)
        kb_max_out = np.max(kb_out)
        kb_range_out = kb_max_out - kb_min_out
        if kb_range_out < _EPS:
            score[outside_mask] = 0.25
        else:
            score[outside_mask] = 0.5 * (kb_out - kb_min_out) / kb_range_out
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
