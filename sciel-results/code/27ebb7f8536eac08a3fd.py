"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_ACQ_START_TIME = None
_DEADLINE = 325.0  # seconds, safety margin before 330s kill
_SAFETY_MARGIN = 5.0


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _ACQ_START_TIME
    if _ACQ_START_TIME is None:
        return False
    elapsed = time.time() - _ACQ_START_TIME
    return elapsed > (_DEADLINE - _SAFETY_MARGIN)


def _matern52_kernel(x1, x2, length_scale=0.2):
    """Compute Matern-5/2 kernel between two 1D arrays (scaled to [0,1])."""
    # x1 shape (n,), x2 shape (m,) -> output (n, m)
    d = np.abs(x1[:, None] - x2[None, :]) / length_scale
    sqrt3d = np.sqrt(3.0) * d
    K = (1.0 + sqrt3d + sqrt3d**2 / 3.0) * np.exp(-sqrt3d)
    return K


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a diversity-penalized Expected Improvement with batch-aware correlation penalty.
    """
    global _ACQ_START_TIME
    _ACQ_START_TIME = time.time()
    
    n = len(mu)
    
    # Check deadline early
    if _check_deadline():
        # Fallback: pure exploitation
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Standard EI
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    ei = sigma_safe * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # For diversity penalty, we need to estimate correlations among top candidates.
    # Since we don't have coordinates, we use the GP posterior structure.
    # The correlation between candidates i and j is approximated by the kernel value
    # at their locations. Without coordinates, we use a proxy: the normalized sigma
    # and mu structure to estimate proximity.
    
    # Alternative: use the fact that in 1D with Matern-5/2, nearby points have high correlation.
    # We can estimate a "diversity score" by looking at how much each point's prediction
    # differs from its neighbors in the sorted-by-mu order.
    
    # Sort by EI to get top candidates
    # Use a diversity penalty based on local density of high-EI points
    
    # Compute a local density estimate: for each point, count how many other points
    # have similar mu and sigma (proxy for being in the same region)
    # This is O(n^2) which might be too slow for n=2000, so we use a binned approach
    
    # Binned approach: bin mu and sigma into a 2D grid
    n_bins = 50
    mu_range = mu.max() - mu.min() if mu.max() > mu.min() else 1.0
    sigma_range = sigma.max() - sigma.min() if sigma.max() > sigma.min() else 1.0
    
    mu_bins = np.clip(np.floor((mu - mu.min()) / mu_range * n_bins).astype(int), 0, n_bins - 1)
    sigma_bins = np.clip(np.floor((sigma - sigma.min()) / sigma_range * n_bins).astype(int), 0, n_bins - 1)
    
    # Count density in each bin
    bin_counts = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(n):
        bin_counts[mu_bins[i], sigma_bins[i]] += 1
    
    # Get local density for each point
    local_density = np.array([bin_counts[mu_bins[i], sigma_bins[i]] for i in range(n)])
    
    # Normalize density to [0, 1]
    max_density = local_density.max() if local_density.max() > 0 else 1.0
    normalized_density = local_density / max_density
    
    # Diversity penalty: penalize points in dense regions
    # The penalty should be proportional to the correlation, which is higher in dense regions
    diversity_penalty = 0.3 * normalized_density * sigma
    
    # Final score: EI minus diversity penalty
    scores = ei - diversity_penalty
    
    # Also add a small UCB component for exploration
    kappa = 0.5
    scores = scores + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline before returning
    if _check_deadline():
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
