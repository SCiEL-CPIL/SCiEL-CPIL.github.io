import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_t0 = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _t0 > _DEADLINE - _MARGIN:
        return np.zeros_like(mu)

    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Standardized improvement z
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-normalized EI
    ei_score = ei / cost_safe

    # Exploration term: sigma normalized by cost
    exploration = 0.3 * sigma_safe / cost_safe

    # Cost bias: prefer costs near 1.5 (center of the domain)
    target_cost = 1.5
    alpha_bias = 1.5
    cost_weight = np.exp(-alpha_bias * (cost_safe - target_cost) ** 2)

    # Combine
    score = (ei_score + exploration) * cost_weight

    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
