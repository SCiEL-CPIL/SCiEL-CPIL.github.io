# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Quantile-Gated Expected Improvement with Uncertainty Rank Selection.
    
    This function implements a structural change: instead of a continuous
    exploration-exploitation trade-off (like LCB or standard EI + bonus),
    it uses a discrete quantile-based gating mechanism.
    
    Strategy:
    1. Compute Expected Improvement (EI) for all candidates.
    2. Rank candidates by uncertainty (sigma).
    3. Select a small top-quantile of high-uncertainty candidates.
    4. Among those, only keep candidates whose mean is below f_best (promising).
    5. For the selected set, score = EI. For others, score = -inf (or very low).
    
    This effectively forces the next evaluation to be one of the most uncertain
    points that are still potentially better than the current best. This is
    structurally different from continuous hybrids because it creates a hard
    selection based on empirical rank, decoupling the exploration decision from
    the specific magnitude of the variance.
    
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
    z_clipped = np.clip(z, -50, 50)
    
    # Use scipy.stats.norm for pdf/cdf
    from scipy.stats import norm
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei_scores = improvement * Phi_z + sigma * phi_z
    
    # 2. Rank-based uncertainty selection
    # Get ranks of sigma (0 = lowest uncertainty, n-1 = highest)
    sigma_ranks = np.argsort(np.argsort(sigma))
    
    # Define the quantile threshold for "high uncertainty"
    # We want the top 5% of candidates by uncertainty
    # So we want ranks >= 0.95 * n
    quantile_threshold = int(0.95 * n)
    
    # 3. Gating: only consider candidates in the top 5% of uncertainty
    # AND whose mean is below f_best (i.e., potentially better)
    high_uncertainty_mask = sigma_ranks >= quantile_threshold
    promising_mask = mu < f_best
    
    # Combined mask: high uncertainty AND promising
    selection_mask = high_uncertainty_mask & promising_mask
    
    # 4. Assign scores
    # For selected candidates, use EI score
    # For others, assign a very low score so they are never chosen
    # We use -1e10 to ensure they are below any valid EI score
    scores = np.full(n, -1e10, dtype=np.float64)
    scores[selection_mask] = ei_scores[selection_mask]
    
    # Edge case: if no candidates are selected, fall back to pure EI
    # This can happen early on when f_best is very high and mu is all above it,
    # or if the uncertainty distribution is degenerate.
    if not np.any(selection_mask):
        # Fallback to standard EI, but maybe with a slight exploration bias
        # If no point is promising, we should explore the most uncertain point
        # that has the lowest mean (best predicted value)
        # Let's just use EI as a fallback, which will pick the best expected improvement
        scores = ei_scores
    else:
        # Ensure finiteness
        scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
