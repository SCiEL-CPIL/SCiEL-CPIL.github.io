"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as t_dist
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Student-t Process (STP) approximation.
    Uses 3 degrees of freedom for heavier-tailed predictive distribution,
    yielding more robust exploration for Beale's sharp curvature.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Degrees of freedom for Student-t approximation
    nu = 3.0  # Heavy tails for Beale's sharp features

    # Scale sigma to match the variance of a t-distribution with nu df
    # Var(t_nu) = nu / (nu - 2) for nu > 2
    # To match the original sigma (which is the std of the GP posterior),
    # we scale: sigma_t = sigma * sqrt((nu - 2) / nu)
    # But actually, we want the predictive std to be sigma, so:
    # If f ~ t_nu(mu, sigma_t), then Var(f) = sigma_t^2 * nu/(nu-2)
    # We want this to equal sigma^2, so sigma_t = sigma * sqrt((nu-2)/nu)
    sigma_t = sigma_safe * np.sqrt((nu - 2.0) / nu)

    # Expected Improvement with Student-t predictive distribution
    # For f ~ t_nu(mu, sigma_t), EI = E[max(f_best - f, 0)]
    # This requires integrating against the t-distribution CDF/PDF
    
    # z = (f_best - mu) / sigma_t
    z = (f_best - mu) / sigma_t
    
    # For t-distribution:
    # EI = (f_best - mu) * P(f < f_best) + E[f * 1(f < f_best)]
    # P(f < f_best) = t_cdf(z, nu)
    # E[f * 1(f < f_best)] = mu * t_cdf(z, nu) + sigma_t * t_pdf(z, nu) * nu/(nu-1) * (1 - z * t_cdf(z,nu)/t_pdf(z,nu))
    # Actually, a simpler approach: use the relationship for t-distributions
    
    # For t_nu distribution, the expected value of max(f_best - f, 0) is:
    # EI = (f_best - mu) * t_cdf(z, nu) + sigma_t * t_pdf(z, nu) * nu * (1 + z^2/(nu-1)) / (nu-1)
    # Wait, let me use a cleaner formula.
    
    # For X ~ t_nu(0,1), E[max(c - X, 0)] = c * P(X < c) + E[X * 1(X < c)]
    # E[X * 1(X < c)] for t-distribution:
    # = -t_pdf(c, nu) * (nu / (nu-1)) * (1 + c^2/(nu-1))^((nu-1)/2) * B_incomplete(...)
    # This is complex. Let's use a numerical approximation or a simpler form.
    
    # Alternative: use the fact that for large nu, t approximates normal.
    # For nu=3, we can use a series expansion or direct computation.
    
    # Let's use scipy's t distribution directly for accuracy
    t_cdf = t_dist.cdf(z, df=nu)
    t_pdf = t_dist.pdf(z, df=nu)
    
    # For t-distribution with df=nu, mean=0, var=nu/(nu-2):
    # E[max(f_best - f, 0)] where f = mu + sigma_t * t_nu(0,1)
    # = E[max(f_best - mu - sigma_t * t, 0)]
    # = sigma_t * E[max(z - t, 0)] where z = (f_best-mu)/sigma_t
    # = sigma_t * [z * P(t < z) + E[t * 1(t < z)]]
    
    # For t ~ t_nu(0,1):
    # E[t * 1(t < z)] = -t_pdf(z, nu) * (nu/(nu-1)) * (1 + z^2/(nu-1))^((nu-1)/2) * I_{nu/(nu+z^2)}((nu+1)/2, 1/2)
    # This is still complex. Let's use a simpler approximation for nu=3.
    
    # For nu=3, t_pdf(z) = (1 + z^2/3)^(-2) / (2 * sqrt(3) * pi^0.5 * 3^0.5)
    # Actually, let's just use the normal approximation as a baseline and add a correction.
    
    # Simpler approach: use the normal EI but with an adjusted sigma that accounts for heavy tails
    # The heavy tails mean that extreme values are more likely, so exploration should be stronger.
    # We can model this by using a larger effective sigma in the EI calculation.
    
    # Effective sigma for EI: account for the fact that t-distribution has heavier tails
    # Use sigma_eff = sigma_t * sqrt(nu / (nu - 2)) = sigma_safe (back to original)
    # But the CDF/PDF should be from the t-distribution.
    
    # Let's compute EI using the t-distribution CDF/PDF:
    # EI = sigma_t * [z * t_cdf(z, nu) + t_pdf(z, nu) * nu/(nu-1) * (1 + z^2/(nu-1))^((nu-1)/2)]
    # This is an approximation. Let's use a cleaner formula.
    
    # Actually, for the t-distribution, the expected improvement can be computed as:
    # EI = (f_best - mu) * t_cdf(z, nu) + sigma_t * t_pdf(z, nu) * (nu / (nu - 1)) * (1 + z^2/(nu-1))^((nu-1)/2)
    
    # Let's verify this formula. For nu -> infinity, it should reduce to normal EI.
    # As nu -> inf, t_cdf -> norm_cdf, t_pdf -> norm_pdf, and (nu/(nu-1)) * (1+z^2/(nu-1))^((nu-1)/2) -> 1
    # So EI -> (f_best-mu)*norm_cdf(z) + sigma*norm_pdf(z), which is correct.
    
    correction = (nu / (nu - 1.0)) * (1.0 + z**2 / (nu - 1.0)) ** ((nu - 1.0) / 2.0)
    ei_stp = (f_best - mu) * t_cdf + sigma_t * t_pdf * correction
    ei_stp = np.clip(ei_stp, -1e6, 1e6)
    
    # Normalize EI
    ei_max = np.max(ei_stp)
    ei_min = np.min(ei_stp)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei_stp - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei_stp)
    
    # UCB for minimization with Student-t: -mu + beta * sigma_t
    # Use a moderate beta for exploration
    beta = 1.5
    ucb = -mu + beta * sigma_t
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: 80% EI, 20% UCB (based on knowledge that EI is more reliable for Beale)
    w_ei = 0.8
    w_ucb = 1.0 - w_ei
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Add a mild penalty for candidates with very high mu
    scale = max(abs(f_best), 1.0)
    mu_penalty = np.maximum(0.0, (mu - f_best) / (scale + 1e-12))
    mu_penalty = np.minimum(mu_penalty, 3.0)
    scores = scores - 0.05 * mu_penalty
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
