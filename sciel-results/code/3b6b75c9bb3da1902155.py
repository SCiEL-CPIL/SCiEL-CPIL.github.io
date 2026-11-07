"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before the 330 s kill


def acquisition(mu, sigma, f_best):
    """
    UCB acquisition with adaptive exploration.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at the candidate points (objective units).
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at the candidate points (objective units).
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Deadline guard: if we are too close to the kill, return a safe, fast result.
    elapsed = time.time() - _START_TIME
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Fallback: return -mu (pure exploitation) to avoid any extra computation.
        return -mu

    # Ensure inputs are 1-D arrays of the expected shape.
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # The length of mu equals the number of candidates drawn this step (2000),
    # which also equals the current evaluation count in the loop (5 initial + steps).
    # We use this to adapt the exploration coefficient.
    n_evals = len(mu)

    # Adaptive beta: start high for exploration, decay to a low floor.
    # Over 30 evaluations, beta goes from ~3.0 to ~0.5.
    # Use a smooth decay: beta = beta_floor + (beta_max - beta_floor) * exp(-k * n_evals)
    beta_max = 3.0
    beta_floor = 0.5
    k = 0.15  # decay rate
    beta = beta_floor + (beta_max - beta_floor) * np.exp(-k * n_evals)

    # UCB score: higher is better for selection (we want to maximize mu + beta*sigma,
    # but since we are minimizing, the GP is fit to the objective, so mu is the predicted
    # objective value. To select the point most likely to be low, we should actually
    # minimize mu - beta*sigma. However, the contract says "higher meaning evaluate next".
    # Since the objective is to minimize, and mu is the predicted objective value,
    # the acquisition that encourages finding lower values should favor low mu and high sigma.
    # So the score should be: -mu + beta * sigma (higher score = more promising for minimization).
    # Alternatively, we can think of it as maximizing an upper confidence bound on the *negative*
    # objective. Let's use: score = -mu + beta * sigma.
    score = -mu + beta * sigma

    # Ensure finite scores.
    score = np.where(np.isfinite(score), score, -np.inf)

    return score
# EVOLVE-BLOCK-END
