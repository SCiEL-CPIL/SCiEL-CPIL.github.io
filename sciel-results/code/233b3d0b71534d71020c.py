"""Cost-aware acquisition for rastrigin-2d BO; EVOLVE-BLOCK rewritten from scratch."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Fixed wall-clock safety margin: the evaluator kills at 330 s; return early if we
# approach that. We do NOT use this to add search; it is a hard guard only.
_DEADLINE_SECONDS = 300.0
_START_TIME = time.time()

# Rastrigin-2d on [-5.12, 5.12]^2; minimizer value is 0.
# Cost of evaluating x is 1 + mean(x_scaled_to_[0,1]).
# We are given mu, sigma, f_best, cost all of shape (2000,). No coordinates.
# We must return 2000 finite scores (higher = evaluate next).


def _safe_return(scores, fallback):
    """Ensure scores are finite and shape (2000,)."""
    if scores is None:
        return fallback
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2000,):
        return fallback
    if not np.all(np.isfinite(scores)):
        # Replace non-finite with a large negative to keep finite but low priority
        scores = np.where(np.isfinite(scores), scores, -1e18)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining exploitation (EI-like) and exploration (UCB-like),
    with adaptive weighting based on how close f_best is to the known minimum (0).
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Higher scores are preferred for the next evaluation.
    """
    # Hard deadline guard
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        # Return a safe, finite fallback: prefer low cost and low mu
        fallback = -mu / (cost + 1e-9)
        return _safe_return(fallback, np.zeros(2000, dtype=np.float64))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Ensure cost is positive (should be >= 1 by construction)
    cost = np.maximum(cost, 1e-9)

    # --- Exploitation: Expected Improvement (EI) ---
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Clip to non-negative
    ei = np.maximum(ei, 0.0)

    # --- Exploration: UCB-like bonus ---
    # UCB = mu - kappa * sigma  (lower is better for minimization, so we use -mu + kappa*sigma)
    # We want higher score = more promising, so:
    # exploration_score = -mu + kappa * sigma
    # Adaptive kappa: larger when f_best is far from 0 (early), smaller when close (late)
    # f_best for rastrigin is >= 0; distance from minimum (0) is f_best itself
    # Use a smooth transition: kappa = kappa_max * exp(-f_best / scale)
    kappa_max = 3.0
    scale = 5.0  # tuning: how quickly exploration decays
    kappa = kappa_max * np.exp(-f_best / scale)

    exploration = -mu + kappa * sigma

    # --- Cost normalization ---
    # Divide by cost to prefer cheaper evaluations
    # We use a soft cost penalty: score / (cost^alpha) where alpha controls cost sensitivity
    # alpha = 1.0 means linear cost penalty
    alpha = 1.0
    cost_factor = cost ** alpha

    # --- Adaptive weighting between EI and exploration ---
    # When f_best is large (early), weight exploration more
    # When f_best is small (late), weight exploitation more
    # weight_exploit = sigmoid((f_best - threshold) / width)
    threshold = 2.0
    width = 2.0
    weight_exploit = 1.0 / (1.0 + np.exp(-(f_best - threshold) / width))
    weight_explore = 1.0 - weight_exploit

    # Combine: higher is better
    # EI is already in "improvement" units (higher = better improvement)
    # Exploration: -mu + kappa*sigma, higher = more promising
    # Scale EI to be comparable to exploration
    # EI is typically O(1) when f_best is O(1); exploration is O(f_best)
    # Normalize EI by a factor to make it comparable
    ei_scale = 1.0
    combined = (weight_exploit * ei * ei_scale + weight_explore * exploration) / cost_factor

    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, -1e18)

    return _safe_return(combined, np.zeros(2000, dtype=np.float64))
# EVOLVE-BLOCK-END
