"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf, ndtr
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on griewank 2d.
    Blends Probability of Improvement with Expected Improvement,
    applies an exponential cost penalty, and includes a shrinking UCB term.
    """
    t0 = time.time()

    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement: P(f(x) < f_best)
    pi_prob = ndtr(z)

    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = (f_best - mu) * pi_prob + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)

    # Normalize EI to be comparable to PI (which is in [0,1])
    # Use a soft normalization based on the scale of EI values
    ei_scale = np.mean(np.abs(ei)) + 1e-9
    ei_norm = ei / ei_scale

    # Blend PI and EI: weight shifts toward EI as we find better solutions
    # When f_best is close to 0, we want more exploitation (EI)
    # When f_best is large, we want more exploration (PI)
    blend_weight = np.clip(1.0 - f_best / (1.0 + f_best), 0.3, 0.7)
    blended = blend_weight * ei_norm + (1.0 - blend_weight) * pi_prob

    # UCB exploration term that shrinks as f_best approaches 0
    beta = 0.1 * (1.0 + f_best)  # exploration coefficient
    ucb_term = beta * sigma_safe / (np.mean(sigma_safe) + 1e-9)

    # Exponential cost penalty
    # Cost is in [1, 2] roughly
    penalty = np.exp(-5.0 * (cost_safe - 1.0))

    score = (blended + ucb_term) * penalty

    # Clamp non-finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Deadline guard
    if time.time() - t0 > 280:
        return score

    return score
# EVOLVE-BLOCK-END
