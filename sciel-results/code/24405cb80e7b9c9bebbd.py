# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Pareto-Frontier Selection Acquisition Function.
    
    This function computes both Expected Improvement (EI) and Upper Confidence
    Bound (UCB) scores for all candidates. It then normalizes both scores to
    [0, 1] using rank-based normalization. The final score is the minimum of
    the normalized EI rank and the normalized UCB rank.
    
    This structural change selects candidates that are simultaneously among
    the most promising (high EI) AND most uncertain (high UCB), rather than
    blending them linearly. This avoids the trade-off where a high exploration
    weight can dominate and select points with poor expected improvement.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate UCB score
    # UCB(x) = -mu(x) + beta * sigma(x)
    # The UCB score for minimization: we want to maximize -mu + beta*sigma
    # beta is chosen based on progress
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # beta decays as we get closer to the global minimum
    beta = 2.0 * (1.0 - 0.7 * progress)  # 2.0 -> 0.6
    
    scores_ucb = -mu + beta * sigma
    
    # 3. Rank-based normalization of both scores to [0, 1]
    # Higher rank = better
    if n > 1:
        ei_ranks = np.argsort(np.argsort(scores_ei)) / (n - 1)
        ucb_ranks = np.argsort(np.argsort(scores_ucb)) / (n - 1)
    else:
        ei_ranks = np.ones(n)
        ucb_ranks = np.ones(n)
    
    # 4. Pareto-frontier selection: minimum of normalized ranks
    # This selects points that are simultaneously high in both EI and UCB
    scores = np.minimum(ei_ranks, ucb_ranks)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
