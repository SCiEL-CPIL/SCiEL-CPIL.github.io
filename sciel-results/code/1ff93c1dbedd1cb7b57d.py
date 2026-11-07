"""Thompson Sampling acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 330.0
_MARGIN = 5.0
_start = time.time()


def acquisition(mu, sigma, f_best):
    # Thompson sampling: draw one sample from the GP posterior and prefer the
    # candidate with the lowest sampled value. Stochastic sampling gives an
    # implicit exploration-exploitation balance that handles the multimodal
    # Michalewicz landscape without hand-tuned xi or UCB constants.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Ensure sigma is a strictly positive standard deviation.
    sigma = np.where(sigma > 1e-9, sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)

    # Draw a single posterior sample (mu ~ N(0, sigma^2) in objective units).
    sample = mu + sigma * np.random.randn(mu.shape[0])

    # Score: higher means "evaluate next". We minimize, so a lower sampled
    # value is better -> score = f_best - sample.
    scores = f_best - sample

    # Guarantee finiteness (the contract rejects non-finite scores).
    scores = np.where(np.isfinite(scores), scores, -1e12)

    # Deadline guard: if we are running out of wall-clock time, still return a
    # valid finite vector (the computation above is O(n) and fast, but guard
    # against pathological inputs).
    if time.time() > _DEADLINE - _MARGIN:
        return scores

    return scores
# EVOLVE-BLOCK-END
