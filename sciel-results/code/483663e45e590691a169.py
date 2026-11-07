"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function:
      score = [ EI_minimisation * (1 + kappa * U ) ] / cost
    where
      EI = (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu) / sigma
      U  = uncertainty bonus  = sigma * Phi(-z)   (exploration in unexplored regions)
      kappa controls how much we weight exploration relative to exploitation.
    """
    # --- guard against zero / negative sigma ---
    sigma_safe = np.maximum(sigma, 1e-9)

    # --- standardised distance ---
    z = (f_best - mu) / sigma_safe

    # --- Expected Improvement for minimisation ---
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # --- Exploration bonus: high where z is very negative (far above f_best) ---
    # Phi(-z) is large when z << 0, i.e. mu >> f_best (unexplored / uncertain region)
    u_bonus = sigma_safe * norm.cdf(-z)

    # --- combine ---
    kappa = 0.5
    combined = ei + kappa * u_bonus

    # --- normalise by cost (cost in [1,5]) ---
    score = combined / cost

    # --- ensure finite ---
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
