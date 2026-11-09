"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge-Gradient-inspired, cost-aware acquisition for Griewank 2D.
    
    KG view: the value of evaluating x is the expected reduction in the
    best observed value, E[max(f_best - Y, 0)] for Y ~ N(mu, sigma),
    divided by the evaluation cost. We augment this with an exploration
    term that rewards candidates with a meaningful probability of being
    a new global minimum, which helps navigate the flat Griewank landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero/negative cost and zero sigma.
    cost_safe = np.where(cost > 1e-12, cost, 1.0)
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)

    # Standardized distance from the current best.
    z = (f_best - mu) / sigma_safe

    # KG / EI term: expected reduction in f_best if we evaluate here.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Exploration term: variance-based exploration to navigate flat landscapes.
    # This decouples the exploration drive from the magnitude of f_best,
    # ensuring we continue to explore high-uncertainty regions even as
    # f_best approaches 0 and EI values shrink.
    exploration_weight = 1.5
    score = (ei + exploration_weight * sigma_safe) / cost_safe

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
