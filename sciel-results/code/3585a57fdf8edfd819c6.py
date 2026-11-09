"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Matern-5/2 kernel with length scale 0.2 on [0,1]^2
_LAMBDA = 5.0  # sqrt(5)/l = sqrt(5)/0.2 = 5*sqrt(5) ≈ 11.18, but let's use standard form
_L = 0.2       # length scale
_SQRT5 = np.sqrt(5.0)

def _matern52_kernel(d2):
    """Matern-5/2 kernel from squared distances."""
    d = np.sqrt(d2 + 1e-15)
    return np.exp(-_SQRT5 * d / _L) * (1.0 + _SQRT5 * d / _L + 5.0 * d2 / (3.0 * _L * _L))

# We don't know the exact candidate coordinates, but we can infer them from
# the GP posterior structure. Since we only receive mu and sigma (not X),
# we need a different structural approach.

# Structural change: Use a "lookahead" acquisition that penalizes candidates
# whose predicted improvement is likely redundant with other high-scoring candidates.
# Without coordinates, we approximate diversity by looking at the distribution of
# mu values: candidates with very similar mu AND high sigma are likely close together.

# Alternative structural approach: Use a modified acquisition that incorporates
# a "coverage" bonus based on how spread out the candidate's posterior parameters
# are relative to the rest of the batch. This encourages exploration in regions
# where the posterior is uncertain AND different from other candidates.

# Key insight: Without coordinates, we can still use the relationship between
# mu and sigma to create a diversity-aware score. Candidates in flat regions
# (low sigma) near f_best get high EI but are redundant. Candidates in uncertain
# regions (high sigma) far from f_best may be exploratory.

# Structural change: Use "expected improvement with adaptive gamma" where
# gamma is estimated from the batch statistics, creating a self-tuning
# exploration-exploitation balance that adapts to the current posterior state.

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    # Base EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural: Compute a "posterior surprise" score that rewards candidates
    # in regions where the GP is uncertain relative to the best value found.
    # This creates a natural exploration bonus that scales with uncertainty.
    #
    # Key structural difference from vanilla EI: we add a term that depends on
    # the *relative* uncertainty across the batch, not just per-candidate sigma.
    # This makes the acquisition batch-aware in a coordinate-free way.
    
    # Normalize sigma to get relative uncertainty
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    rel_sigma = (sigma - sigma_mean) / sigma_std
    
    # "Uncertainty bonus": reward candidates that are more uncertain than average,
    # but scale it by how much room there is to improve (f_best - mu)
    # This creates a preference for exploring uncertain regions that could beat f_best
    improvement_headroom = np.maximum(f_best - mu, 0.0)
    uncertainty_bonus = rel_sigma * improvement_headroom * 0.5
    
    # "Diversity penalty": penalize candidates whose (mu, sigma) are very similar
    # to other high-EI candidates. Approximate by looking at the distribution
    # of mu values among the top-scoring candidates.
    #
    # We use a soft penalty: for each candidate, compute how many other candidates
    # have mu within a small band. More neighbors = more redundancy.
    # This is O(n^2) but n=2000 so it's 4M operations, acceptable.
    
    # Use a histogram-based approach for efficiency
    mu_min, mu_max = np.min(mu), np.max(mu)
    if mu_max - mu_min < 1e-12:
        density_penalty = np.zeros(n)
    else:
        # Create a fine grid to estimate local density
        n_bins = 200
        counts, bin_edges = np.histogram(mu, bins=n_bins)
        bin_width = (mu_max - mu_min) / n_bins
        # For each candidate, find its bin and use the count as density estimate
        bin_indices = np.digitize(mu, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        local_density = counts[bin_indices]
        # Normalize density
        density_penalty = local_density / (local_density.max() + 1e-9)
    
    # Combine: EI + uncertainty bonus - diversity penalty
    # The diversity penalty is scaled so it only matters when EI is high
    # (we don't want to penalize low-EI candidates)
    ei_norm = ei / (ei.max() + 1e-9)
    penalty_weight = 0.3 * ei_norm
    
    score = ei + uncertainty_bonus - penalty_weight * density_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
