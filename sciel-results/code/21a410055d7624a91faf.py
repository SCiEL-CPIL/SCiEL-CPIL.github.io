import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with a small exploration bonus.
    # The 0.05 * sigma term was measured to improve score from 1.0 to 1.264.
    # We keep that design and tune the bonus slightly, adding a guard.
    #
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / (sigma + 1e-9)

    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Exploration bonus: encourages sampling where uncertainty is high.
    # 0.05 was the measured improvement; try a small bump to 0.08.
    score = (ei + 0.08 * sigma) / (cost + 1e-9)

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
