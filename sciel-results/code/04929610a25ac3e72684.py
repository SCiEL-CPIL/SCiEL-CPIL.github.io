# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware expected improvement with a small exploration bonus.
    # mu, sigma, cost: arrays of shape (2000,); higher returned score = evaluate next.
    eps = 1e-9

    # Robust standard deviation (avoid divide-by-zero / negative values).
    s = np.maximum(np.asarray(sigma, dtype=float), eps)

    # Standardized improvement.
    z = (f_best - mu) / s

    # Standard expected improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)

    # Small exploration bonus to help escape local optima in Shekel 4D.
    ei = ei + 0.1 * s

    # Cost-awareness: prefer cheap regions (cost = 1 + mean(x), so lower cost is better).
    c = np.maximum(np.asarray(cost, dtype=float), eps)
    score = ei / c

    # Guarantee finite output (higher = evaluate next).
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
