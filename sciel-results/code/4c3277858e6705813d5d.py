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
    p_new_best = norm.cdf(z)

    # Exploration bonus: reward candidates where discovery is plausible
    # but not yet certain, to navigate flat regions. We use a peaked
    # function of p_new_best that is maximized at moderate probabilities.
    # This encourages exploring points that are "close" to the best but
    # not obviously dominated.
    explore = p_new_best * (1.0 - p_new_best)

    # Combine KG value and exploration, weighted.
    # The EI term is in objective units; the exploration term is dimensionless.
    # Scale exploration to be comparable to EI.
    scale = np.abs(f_best) + 1e-6
    score = ei / cost_safe + 0.1 * scale * explore / cost_safe

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
