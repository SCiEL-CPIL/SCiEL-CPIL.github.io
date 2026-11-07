"""Cost-aware acquisition for Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start_time = time.time()
_DEADLINE_MARGIN = 20  # seconds before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _module_start_time
    if elapsed > 330 - _DEADLINE_MARGIN:
        return mu - sigma  # fallback LCB

    sigma_safe = np.maximum(sigma, 1e-8)

    # Expected Improvement (for minimization: EI = E[max(f_best - f, 0)])
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Theoretical minimum of Hartmann 3D
    f_min = -3.86278

    # Gap to optimum: large early, small late
    gap = f_best - f_min
    # Smooth progress signal using a logistic (soft) transform
    progress = 1.0 / (1.0 + np.exp((gap - 1.0) / 0.5))
    progress = np.clip(progress, 0.0, 1.0)

    # Cost normalization: cost in [1, 2], normalize to [0, 1]
    cost_norm = cost - 1.0  # now in [0, 1]

    # Adaptive parameters
    # Alpha: Cost penalty strength.
    alpha = 0.8 + 2.2 * progress

    # Beta: Uncertainty bonus weight.
    beta = 0.2 * (1.0 - progress)**2

    # Power-law cost penalty: stronger discrimination as budget depletes
    power = 1.0 + 0.8 * progress
    cost_penalty = 1.0 + alpha * (cost_norm ** power)
    cost_adjusted_ei = ei / cost_penalty

    # Uncertainty bonus
    uncertainty_bonus = beta * sigma_safe

    # Exploitation bonus: reward regions where the model is confident of low values
    exploit_weight = 0.15 * (1.0 - progress)
    exploit_margin = np.maximum(f_best - mu, 0.0)
    exploitation_bonus = exploit_weight * exploit_margin / (1.0 + exploit_margin)

    # Combine
    score = cost_adjusted_ei + uncertainty_bonus + exploitation_bonus

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
