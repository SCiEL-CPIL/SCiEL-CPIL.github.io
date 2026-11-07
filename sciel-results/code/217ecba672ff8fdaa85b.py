# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # mu, sigma: shape (2000,) in objective units; f_best: lowest observed value
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    # Guard against non-positive / degenerate sigma
    sigma = np.maximum(sigma, 1e-9)

    # Expected improvement (minimization): how much we expect to beat f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB exploration term
    beta = 0.5
    ucb = beta * sigma

    # Blend: as the best value approaches the known minimum (0), the EI term
    # becomes more reliable and we lean on it; early on the UCB term encourages
    # exploring high-uncertainty regions.
    # scale factor in [0,1]: 1 when f_best is far from 0, 0 when f_best ~ 0
    scale = min(1.0, max(0.0, f_best))
    # weight for EI grows as we get close to the optimum
    w_ei = 0.5 + 0.5 * scale
    w_ucb = 1.0 - w_ei

    score = w_ei * ei + w_ucb * ucb

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
