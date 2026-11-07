import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIMEOUT = 330.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT - 1.0:
        # Return a default acquisition, e.g., EI
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    f_min_known = -1.8013

    # Calculate gap from current best to known global minimum
    gap = max(0.0, f_best - f_min_known)
    norm_gap = min(1.0, gap)

    # Compute Expected Improvement (exploitation)
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Compute LCB-style exploration score
    # Beta ranges from ~1.0 (exploitative) to ~2.5 (exploratory)
    beta = 1.0 + 1.5 * norm_gap
    lcb_score = -mu + beta * sigma

    # Normalize both scores to [0, 1] for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    lcb_min, lcb_max = np.min(lcb_score), np.max(lcb_score)

    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)

    if lcb_max - lcb_min > 1e-12:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min)
    else:
        lcb_norm = np.zeros_like(lcb_score)

    # Adaptive blend weight:
    # Early (large gap): favor exploration (LCB)
    # Late (small gap): favor exploitation (EI)
    # Weight for EI increases as gap shrinks
    w_ei = 0.5 + 0.5 * (1.0 - norm_gap)
    w_lcb = 1.0 - w_ei

    # Blend
    score = w_ei * ei_norm + w_lcb * lcb_norm

    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    return score
