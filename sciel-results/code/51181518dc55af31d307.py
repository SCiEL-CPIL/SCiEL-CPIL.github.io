import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0  # Safety margin well below the 330 s wall-clock kill

# Evaluation budget: 5 initial + 25 BO steps = 30 total
_TOTAL_EVALS = 30
_call_counter = 0

def acquisition(mu, sigma, f_best):
    """
    Dynamic Lower Confidence Bound (LCB) acquisition for minimizing the objective.

    The score is defined as: Score = beta * sigma - mu
    Higher scores indicate candidates that are more likely to yield a new minimum,
    balancing exploitation (low mu) and exploration (high sigma).

    beta is dynamically adjusted based on the evaluation budget: larger early
    (explore) and smaller late (exploit). We track the number of acquisition
    calls as a proxy for the evaluation step index.

    mu, sigma : array-like, shape (2000,)  (objective units, sigma >= 0)
    f_best    : float, lowest value observed so far

    Returns: array of 2000 finite scores; HIGHER means "evaluate next".
    """
    global _call_counter
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # ---- wall-clock guard: never let this call approach the kill limit ----
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Exploitation-only fallback (finite, fast).
        score = -mu
        return np.clip(score, -1e12, 1e12)

    # Handle potential numerical issues in sigma
    # Use a log-space floor to avoid over-exploring zero-noise candidates
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Track call index (0-based). First 5 calls correspond to initial random points
    # (acquisition may still be called), then 25 BO steps.
    idx = _call_counter
    _call_counter += 1

    # Budget-based beta: start high (explore), decay to low (exploit)
    # At idx=0 (first call): progress = 0 -> beta ~ beta_max
    # At idx=29 (last call): progress = 1 -> beta ~ beta_min
    # Use a smooth decay: beta = beta_min + (beta_max - beta_min) * (1 - progress)^p
    # With p=1.5 for slightly faster decay early.
    progress = min(1.0, max(0.0, idx / max(1, _TOTAL_EVALS - 1)))
    beta_min = 1.5
    beta_max = 4.0
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress) ** 1.5

    # LCB score for minimization: minimize (mu - beta * sigma)
    # Since we return scores where higher is better:
    # Score = -(mu - beta * sigma) = beta * sigma - mu
    score = beta * sigma_safe - mu

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    # Clip to prevent overflow issues in downstream comparisons if any
    score = np.clip(score, -1e12, 1e12)

    # Guarantee exactly `n` finite entries (defensive).
    if score.shape[0] != n:
        score = np.resize(score, n)

    return score
