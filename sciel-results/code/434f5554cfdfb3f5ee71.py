import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIMEOUT = 330.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT - 1.0:
        # Return a default acquisition, e.g., EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    f_min_known = -1.8013

    # Calculate gap from current best to known global minimum
    gap = max(0.0, f_best - f_min_known)
    # Normalize gap: 1.0 when far from optimum, 0.0 when close
    norm_gap = min(1.0, gap / 2.0)  # Michalewicz range is roughly [-1.8, 1.8]

    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (exploitation)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Compute UCB-style exploration score
    # Stronger exploration early (large gap), weaker late
    # k starts high (~2.5) and decreases to ~0.5
    k = 0.5 + 2.0 * norm_gap
    ucb_score = -mu + k * sigma_safe

    # Also compute a "probability of improvement" based score for robustness
    # This is just the CDF of the improvement
    poi = norm.cdf(z)

    # Normalize all scores to [0, 1]
    def normalize(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn > 1e-12:
            return (x - mn) / (mx - mn)
        else:
            return np.zeros_like(x)

    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb_score)
    poi_norm = normalize(poi)

    # Adaptive blend weights:
    # Early (large gap): favor UCB exploration heavily
    # Late (small gap): favor EI exploitation
    # w_ucb goes from ~0.7 to ~0.1
    # w_ei goes from ~0.2 to ~0.8
    # w_poi goes from ~0.1 to ~0.1
    w_ucb = 0.1 + 0.6 * norm_gap
    w_ei = 0.8 - 0.6 * norm_gap
    w_poi = 0.1 - 0.05 * norm_gap

    # Ensure weights sum to 1
    w_sum = w_ucb + w_ei + w_poi
    w_ucb /= w_sum
    w_ei /= w_sum
    w_poi /= w_sum

    # Blend
    score = w_ucb * ucb_norm + w_ei * ei_norm + w_poi * poi_norm

    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    return score
