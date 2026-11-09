import numpy as np


def acquisition(mu, sigma, f_best, cost):
    cost_safe = np.maximum(cost, 1e-9)
    sqrt_cost = np.sqrt(cost_safe)
    kappa = 2.0
    score = -mu + kappa * sigma / sqrt_cost
    return np.where(np.isfinite(score), score, 0.0)
