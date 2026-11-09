"""Cost-aware acquisition function with smooth adaptive parameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)


def _compute_ucb(mu, sigma, f_best, kappa):
    """Compute Upper Confidence Bound for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    ucb = f_best - mu + kappa * sigma_safe
    return np.where(np.isfinite(ucb), ucb, 0.0)


def _smooth_transition(x, low, high):
    """Smooth sigmoid-like transition from 0 to 1 as x goes from low to high."""
    if high <= low:
        return 1.0 if x >= high else 0.0
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    # Smoothstep: 3t^2 - 2t^3
    return 3.0 * t * t - 2.0 * t * t * t


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses a smooth adaptive blend of Expected Improvement and UCB, scaled by cost.
    Parameters are tuned based on the current optimization state using continuous
    functions of f_best and uncertainty to avoid discontinuous behavior.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute base acquisition components
    ei = _compute_ei(mu, sigma, f_best)
    
    # Smooth adaptive kappa: continuous function of f_best and uncertainty
    sigma_mean = np.mean(sigma_safe)
    
    # Base kappa decreases smoothly as f_best improves (lower is better)
    # At f_best ~ 0 (optimal), kappa ~ 0.5 (exploit)
    # At f_best ~ 3+ (far from optimal), kappa ~ 2.0 (explore)
    # Use smooth transition: kappa_base = 0.5 + 1.5 * (1 - smooth_transition(f_best, 0, 3))
    t_exploit = _smooth_transition(f_best, 0.0, 3.0)
    kappa_base = 0.5 + 1.5 * (1.0 - t_exploit)
    
    # Scale by uncertainty: higher uncertainty -> more exploration
    if sigma_mean > 1e-9:
        # Uncertainty factor: 0.7 when sigma_mean is small, 1.3 when large
        unc_factor = 0.7 + 0.6 * _smooth_transition(sigma_mean, 0.01, 0.5)
        kappa = kappa_base * unc_factor
    else:
        kappa = kappa_base
    
    # Clamp kappa
    kappa = float(np.clip(kappa, 0.3, 3.0))
    
    ucb = _compute_ucb(mu, sigma, f_best, kappa)
    
    # Smooth adaptive blending weight
    # t_exploit: 0 when f_best is small (exploit EI), 1 when f_best is large (explore UCB)
    # w_ei = 1 - 0.5 * t_exploit  (ranges from 1.0 to 0.5)
    # w_ucb = 0.5 * t_exploit     (ranges from 0.0 to 0.5)
    # This keeps EI always contributing at least 50%
    w_ei = 1.0 - 0.5 * t_exploit
    w_ucb = 0.5 * t_exploit
    
    # Combine
    combined = w_ei * ei + w_ucb * ucb
    
    # Cost-aware scaling: cost = 1 + mean(x_scaled), so cost ∈ [1, 2]
    # Use linear scaling: divide by (1 + alpha * (cost - 1)) where alpha controls sensitivity
    # alpha = 1.0 means full cost sensitivity; alpha = 0.5 means half sensitivity
    # Smoothly reduce cost sensitivity as f_best improves (we're closer to done)
    alpha = 1.0 - 0.5 * (1.0 - t_exploit)  # alpha ranges from 0.5 to 1.0
    cost_scaled = 1.0 + alpha * (cost - 1.0)
    cost_scaled = np.maximum(cost_scaled, 1e-6)
    
    total_score = combined / cost_scaled
    
    # Ensure finiteness
    total_score = np.where(np.isfinite(total_score), total_score, -1e10)
    
    # Return raw scores (no normalization to preserve relative magnitudes)
    return total_score
# EVOLVE-BLOCK-END
