"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Time guard: kill before the 330s wall-clock limit.
_DEADLINE = time.time() + 325.0


def _time_left():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best):
    # Expected improvement with a time-aware exploration length scale.
    # As the wall-clock budget is consumed we shift toward exploitation
    # by shrinking the effective uncertainty, so late evaluations focus
    # on refining the best region instead of wasting points exploring.
    tl = _time_left()
    if tl <= 0:
        # Safety margin exceeded: fall back to pure exploitation.
        return -(mu - f_best)

    # Fraction of the 330s budget already spent (0 at start, 1 at end).
    frac = 1.0 - max(tl, 0.0) / 330.0
    # Exploration weight: 1.0 early, decays toward ~0.25 late.
    w = 1.0 - 0.75 * frac

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe

    # Standard EI.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Blend with a pure-exploitation score so late search converges.
    exploit = f_best - mu
    scores = w * ei + (1.0 - w) * exploit

    # Ensure finiteness; non-finite values invalidate the whole result.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
