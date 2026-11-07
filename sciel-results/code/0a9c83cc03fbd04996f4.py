"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Student-t Expected Improvement.
    
    mu, sigma, f_best, cost: 1-D arrays of shape (2000,).
    Returns 2000 finite scores; higher = evaluate next.
    
    Strategy:
    - Use Student-t distribution with nu=3 degrees of freedom to account for
      model uncertainty (heavier tails than Gaussian EI).
    - Inflate sigma slightly to encourage exploration in uncertain regions.
    - Scale by inverse cost with power alpha=0.8 (less aggressive than linear).
    - Add confidence-adaptive exploitation: when f_best is very good (low regret),
      reduce exploration to exploit.
    """
    # Ensure sigma is positive
    sig = np.maximum(sigma, 1e-9)
    
    # Inflate sigma to capture heavier-tailed model uncertainty
    # This makes the acquisition more robust to GP misspecification
    sigma_inflated = sig * 1.2
    
    # Calculate z-score for Student-t distribution
    # For Student-t with nu df, the CDF and PDF are different from normal
    z = (f_best - mu) / sigma_inflated
    z = np.clip(z, -10.0, 10.0)
    
    # Student-t Expected Improvement
    # EI_t = E[max(f_best - f(x), 0)] where f(x) ~ t(mu, sigma, nu)
    # For nu=3, we can compute this using the Student-t CDF and PDF
    nu = 3.0
    
    # Student-t CDF and PDF
    # Using scipy.stats.t
    t_cdf = student_t.cdf(z, df=nu)
    t_pdf = student_t.pdf(z, df=nu)
    
    # Expected Improvement with Student-t
    # The formula is similar but with t-distribution moments
    # E[max(a - X, 0)] where X ~ t(mu, sigma, nu)
    # = (f_best - mu) * P(X < f_best) + sigma * nu * t_pdf(z) * (1 + z^2/nu) / (nu-1) for nu > 1
    # More precisely, for Student-t:
    # EI = (f_best - mu) * t_cdf(z) + sigma_inflated * t_pdf(z) * (nu * (1 + z^2/nu)) / (nu - 1)
    
    # Simplified: use the standard EI formula but with t-distribution
    # Actually, the correct formula for Student-t EI is:
    # EI = (f_best - mu) * t_cdf(z) + sigma_inflated * t_pdf(z) * nu / (nu - 1) * (1 + z^2/nu)
    
    correction = (nu / (nu - 1)) * (1 + z**2 / nu)
    ei_t = (f_best - mu) * t_cdf + sigma_inflated * t_pdf * correction
    
    # Ensure EI is non-negative
    ei_t = np.maximum(ei_t, 0.0)
    
    # Confidence-adaptive exploitation:
    # If f_best is very good (close to minimum), reduce exploration
    # Estimate how "good" f_best is by looking at the scale of mu
    # When f_best is much lower than typical mu values, we're in exploitation mode
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    f_best_norm = (f_best - np.min(mu)) / mu_range
    # If f_best is in the lower 20% of observed values, increase exploitation
    exploitation_boost = 1.0 + 0.5 * np.maximum(0, 1.0 - f_best_norm)
    
    # Cost-aware scoring with power alpha=0.8
    # Less aggressive than linear (alpha=1.0) to allow reaching higher-cost minima
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** 0.8
    
    # Final score
    score = (ei_t * exploitation_boost) / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
