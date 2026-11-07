# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Regret (ER) acquisition with a smooth exploration bonus.
    ER = P(f < f_best) = Phi((f_best - mu) / sigma).
    We add a small, smooth uncertainty bonus to encourage exploration
    without dominating the probability-of-improvement signal.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)

    # Primary signal: probability of improvement
    scores = norm.cdf(z)

    # Smooth exploration bonus: use a soft-max of sigma to avoid hard
    # dependence on a single outlier. The bonus is scaled so that it
    # contributes at most ~5% to the total score range.
    # We use a log-sum-exp style normalization for smoothness.
    s_max = np.max(s)
    if s_max > 1e-12:
        # Normalize s to [0, 1] range smoothly
        s_norm = s / (s_max + 1e-12)
        # Apply a smooth saturation (tanh-like via sigmoid) to keep the bonus
        # from being dominated by a single extreme sigma value
        s_bonus = np.tanh(s_norm)
        scores += 0.05 * s_bonus

    return scores
# EVOLVE-BLOCK-END
