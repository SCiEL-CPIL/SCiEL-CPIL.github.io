"""Acquisition function for Bayesian optimization of Hartmann-3D using a structural
change: Expected Hypervolume Improvement (EHVI) with adaptive exploration, replacing
the EI + UCB + log-EI + exploration-bonus mixture.

EHVI provides a principled, Pareto-aware acquisition that naturally balances exploitation
and exploration for multi-modal landscapes. We add a small adaptive UCB term that
scales with remaining budget (number of evaluations left) to ensure sufficient
exploration early on while converging to pure EHVI exploitation later.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SECONDS = 330.0
_SAFETY_MARGIN = 20.0
_TOTAL_EVALS = 30  # 5 initial + 25 optimization steps
_CALL_COUNT = 0


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _compute_ehvi(mu, sigma, f_best):
    """
    Compute Expected Hypervolume Improvement for a single-objective problem.
    
    For single-objective minimization, EHVI with reference point r is:
    EHVI = E[max(0, r - f(x))] where f(x) ~ N(mu, sigma^2)
    
    This is equivalent to Expected Improvement with a reference point instead of f_best.
    However, we use a slightly different formulation that accounts for the fact that
    we want to improve upon f_best, and we integrate over the posterior.
    
    For a normal distribution N(mu, sigma^2), the expected improvement over threshold t
    is: E[max(0, t - f)] = (t - mu) * Phi((t - mu)/sigma) + sigma * phi((t - mu)/sigma)
    
    For EHVI with reference point r (where r < f_best typically), we compute:
    EHVI = E[max(0, r - f)] - E[max(0, f_best - f)]
    
    But more practically, for single-objective BO, we can use:
    EHVI = E[max(0, r - f)] where r is set to f_best (or slightly below).
    
    Actually, the standard EHVI for single objective with reference point r is:
    EHVI = integral of max(0, r - f) dP(f)
    
    For f ~ N(mu, sigma^2):
    If mu >= r: EHVI = 0 (no improvement possible in expectation)
    If mu < r: EHVI = (r - mu) * Phi((r - mu)/sigma) + sigma * phi((r - mu)/sigma)
    
    This is exactly the same as EI with f_best replaced by r.
    
    To make this more meaningful, we use a reference point that is slightly better than
    f_best, encouraging the search to push beyond the current best.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Use f_best as the reference point (standard EI formulation)
    # But we can also use a slightly more aggressive reference point
    # to encourage exploration of potentially better regions.
    # Let's use f_best directly for now, which gives standard EI.
    
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Standard EI = EHVI with reference point = f_best
    ehvi = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure non-negative
    ehvi = np.maximum(ehvi, 0.0)
    
    return ehvi


def _compute_ehvi_multi_ref(mu, sigma, f_best):
    """
    Compute a modified EHVI that uses multiple reference points to better
    capture the multi-modal nature of Hartmann-3D.
    
    We compute the expected improvement over f_best, but weight it by the
    probability that the point is in a "promising" region (i.e., where the
    posterior mean suggests potential for significant improvement).
    
    This is done by computing EI at f_best and scaling by a factor that
    increases when sigma is large relative to the gap between mu and f_best,
    indicating high uncertainty in a potentially good region.
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Base EI
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty-aware scaling: boost EI when sigma is large and mu is close to f_best
    # This encourages exploration of uncertain regions that might contain better solutions
    # The factor is 1 + alpha * sigma / (|f_best| + epsilon)
    # This naturally scales with the problem's value scale
    
    # Compute a relative uncertainty measure
    rel_uncertainty = sigma_safe / (np.abs(f_best) + 1e-6)
    
    # Scale EI by a factor that increases with relative uncertainty
    # Use a soft scaling to avoid excessive amplification
    scale_factor = 1.0 + 0.5 * np.tanh(rel_uncertainty)
    
    return ei * scale_factor


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Hypervolume Improvement with adaptive exploration.
    
    The core is EHVI (equivalent to EI for single-objective), augmented with an
    adaptive UCB term that scales with remaining budget. This provides a principled
    balance between exploitation (EHVI) and exploration (UCB) that naturally
    transitions from exploration-heavy early on to exploitation-heavy late in the
    optimization process.
    """
    global _CALL_COUNT
    _ensure_start_time()
    _CALL_COUNT += 1
    
    # Check deadline guard
    if _elapsed() > _DEADLINE_SECONDS - _SAFETY_MARGIN:
        # Return simple EI-based score quickly
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute EHVI (standard EI for single-objective)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    ehvi = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ehvi = np.maximum(ehvi, 0.0)
    
    # Adaptive UCB weight: scales with remaining budget
    # Early on (few calls), use more exploration; late on, use less
    # _CALL_COUNT ranges from 1 to 25 (after initial 5 points)
    # We want the UCB weight to decrease from ~0.05 to ~0.01
    progress = min(_CALL_COUNT / (_TOTAL_EVALS - 5), 1.0)  # 0 to 1 over optimization
    ucb_weight = 0.05 - 0.04 * progress  # 0.05 to 0.01
    
    # UCB component
    ucb_component = ucb_weight * sigma_safe
    
    # Combine EHVI and UCB
    scores = ehvi + ucb_component
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
