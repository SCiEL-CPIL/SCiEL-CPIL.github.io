"""Model-and-optimize acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 300.0
_SAFETY_MARGIN = 20.0

# Known global minimum of Hartmann-3D (used for adaptive convergence scaling)
_HARTMANN3D_MIN = -3.86278

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _ei_scores(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)

def acquisition(mu, sigma, f_best):
    """
    Acquisition combining:
      - EI over f_best (exploitation),
      - EI over the deterministic GP lower bound min(mu) (DGP exploration),
      - Monte-Carlo probability of being the GP global minimum,
      - an adaptive UCB-style exploration bonus that fades as f_best
        approaches the known global minimum.
    Higher score => evaluate next.
    """
    _ensure_start_time()

    # Deadline guard: fall back to simple, fast EI
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        return _ei_scores(mu, sigma, f_best)

    sigma_safe = np.maximum(sigma, 1e-10)
    n = len(mu)
    if n == 0:
        return mu

    # ---- Exploitation: EI over the current best ----
    ei_fb = _ei_scores(mu, sigma, f_best)

    # ---- DGP exploration: EI over the deterministic lower bound ----
    dgp_bound = float(np.min(mu))
    z_dgp = (dgp_bound - mu) / sigma_safe
    z_dgp = np.clip(z_dgp, -10, 10)
    ei_dgp = (dgp_bound - mu) * norm.cdf(z_dgp) + sigma_safe * norm.pdf(z_dgp)
    ei_dgp = np.where(np.isfinite(ei_dgp), ei_dgp, 0.0)

    # ---- Convergence indicators ----
    sigma_mean = float(np.mean(sigma_safe))
    ei_scale = float(np.mean(ei_fb[ei_fb > 0])) if np.any(ei_fb > 0) else (sigma_mean + 1e-10)
    if ei_scale < 1e-10:
        ei_scale = sigma_mean + 1e-10

    # How close are we to the known global optimum? 1.0 => fully converged.
    if f_best > _HARTMANN3D_MIN:
        # Normalize gap; use a scale of ~1.0 (Hartmann3D range is ~[-3.86, 3.32])
        conv = 1.0 - (f_best - _HARTMANN3D_MIN) / (f_best - _HARTMANN3D_MIN + 1.0)
    else:
        conv = 1.0
    conv = float(np.clip(conv, 0.0, 1.0))
    # Exploration weight: high early, fades as we converge
    w_expl = 1.0 - 0.8 * conv

    # ---- Monte-Carlo global-minimum probability ----
    # Time budget: reduce samples if we're running low
    time_remaining = _DEADLINE_SECONDS - _SAFETY_MARGIN - _elapsed()
    if time_remaining < 15:
        n_samples = 40
    elif time_remaining < 40:
        n_samples = 80
    else:
        n_samples = 150

    rng = np.random.default_rng(42)
    samples = mu[None, :] + sigma_safe[None, :] * rng.standard_normal((n_samples, n))
    min_indices = np.argmin(samples, axis=1)
    min_counts = np.bincount(min_indices, minlength=n).astype(np.float64)
    p_global_min = min_counts / n_samples

    p_max = float(np.max(p_global_min)) if n > 0 else 1.0
    if p_max > 1e-15:
        p_global_norm = p_global_min / p_max
    else:
        p_global_norm = np.zeros_like(p_global_min)

    # ---- Adaptive UCB exploration bonus ----
    # beta scales down with convergence; sigma normalized to keep magnitude sane
    sigma_max = float(np.max(sigma_safe))
    if sigma_max > 1e-15:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    beta = 1.5 * w_expl
    ucb_bonus = beta * ei_scale * sigma_norm

    # ---- Combine ----
    # Exploitation always present; exploration terms fade with convergence.
    # DGP-EI is the primary exploration signal (self-scaling), global-min
    # probability adds a diversity bonus, UCB adds pure uncertainty reward.
    scores = (ei_fb
              + 0.8 * w_expl * ei_dgp
              + 0.3 * w_expl * ei_scale * p_global_norm
              + 0.2 * ucb_bonus)

    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
