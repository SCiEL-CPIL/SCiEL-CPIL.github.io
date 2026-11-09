"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    deadline = t0 + 250.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Temperature-scaled EI: adapt exploration/exploitation to progress
    # mu_min is the best predicted value; if f_best is close to mu_min,
    # we're in exploitation mode. If f_best is far, we explore more.
    mu_min = np.min(mu)
    gap = f_best - mu_min
    if gap < 1e-9:
        gap = 1e-9

    # Temperature: high early (more exploration), low late (exploitation)
    # Normalize gap relative to typical sigma to get a meaningful scale
    sigma_med = np.median(sigma)
    if sigma_med < 1e-9:
        sigma_med = 1.0
    norm_gap = gap / sigma_med

    # Temperature decays as we get closer to the optimum
    # temp > 1 means more exploration (wider z distribution), temp < 1 means exploitation
    temp = 1.0 + 0.5 * np.exp(-norm_gap)

    # Scale z by temperature: larger temp -> smaller |z| -> more mass in tails
    z = (f_best - mu) / (sigma * temp)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Progressive cost weighting: reduce cost penalty as optimization progresses
    # When gap is large (early), cost matters more; when gap is small (late),
    # we care less about cost and more about finding the minimum
    cost_weight = 1.0 + 0.5 * np.exp(-norm_gap)
    score = ei / (cost * cost_weight)

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    if time.time() > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
