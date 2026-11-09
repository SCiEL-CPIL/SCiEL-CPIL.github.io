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

    # Probability that this candidate is a new minimum (discovery probability).
    # This decouples the "chance" of improvement from the "magnitude" of improvement.
    p_new_best = norm.cdf(z)

    # Exploration bonus: reward candidates with a high probability of being
    # a new best. This helps navigate flat regions by favoring points that
    # are likely to improve the current best, even if the expected improvement
    # magnitude is small (which happens when uncertainty is high but mean is
    # close to f_best).
    # We use p_new_best directly, scaled by the current best value to match
    # objective units. This weight is increased compared to the previous
    # heuristic to better balance exploration and exploitation.
    scale = np.abs(f_best) + 1e-6
    explore_weight = 0.5
    score = ei / cost_safe + explore_weight * scale * p_new_best / cost_safe

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
