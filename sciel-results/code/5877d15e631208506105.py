"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a Hypervolume-based acquisition that scores candidates by the volume
    of the Pareto front they would add to the set of non-dominated (mu, sigma) points,
    explicitly optimizing for a diverse set of trade-offs.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return safe scores if approaching the 330s limit
    if time.time() - _start_time > 325.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize mu and sigma to [0, 1] for hypervolume computation
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma_safe), np.max(sigma_safe)
    
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # For minimization, we want low mu and low sigma.
    # Pareto dominance: point A dominates point B if A.mu <= B.mu and A.sigma <= B.sigma
    # and at least one is strictly less.
    
    # Compute the hypervolume contribution for each candidate.
    # We use a simplified approach: for each candidate, compute the area in (mu, sigma) space
    # where it is non-dominated and would improve the Pareto front.
    
    # To keep it efficient, we compute for each candidate the "dominated region" volume
    # that it uniquely contributes. We approximate this by considering the 2D grid.
    
    # A practical approach: score each candidate by how much it improves the hypervolume
    # of the non-dominated set. We'll use a sampling-based approximation.
    
    # For efficiency, we'll compute a score based on:
    # 1. How far the candidate is from the ideal point (min mu, min sigma)
    # 2. How much it dominates other points (or is dominated by fewer points)
    
    # Compute dominance count: number of points that dominate this candidate
    # A point i dominates point j if mu_i <= mu_j and sigma_i <= sigma_j and (mu_i < mu_j or sigma_i < sigma_j)
    
    # For efficiency with 2000 points, we can use a vectorized approach
    # but O(n^2) = 4M comparisons which is fine.
    
    # Compute dominance: for each candidate i, count how many j dominate i
    # mu[i] <= mu[j] and sigma[i] <= sigma[j] and (mu[i] < mu[j] or sigma[i] < sigma[j])
    
    # We'll compute a score that rewards points that are non-dominated and close to the ideal.
    
    # Ideal point: (min mu, min sigma)
    ideal_mu = mu_min
    ideal_sigma = sigma_min
    
    # Distance to ideal in normalized space
    dist_to_ideal = np.sqrt((mu_norm - 0.0)**2 + (sigma_norm - 0.0)**2)
    
    # Compute dominance: for each i, how many j have mu_j <= mu_i and sigma_j <= sigma_i
    # and at least one strict
    # This is O(n^2) which is 4M operations, acceptable.
    
    # Vectorized dominance count
    # For each i, count j where mu[j] <= mu[i] and sigma[j] <= sigma[i] and (mu[j] < mu[i] or sigma[j] < sigma[i])
    
    # We can compute this more efficiently using sorting and cumulative counts,
    # but for 2000 points, direct computation is fine.
    
    # Compute dominance count
    # dominance_count[i] = number of j != i such that mu[j] <= mu[i] and sigma[j] <= sigma[i] and (mu[j] < mu[i] or sigma[j] < sigma[i])
    
    # Use broadcasting: mu[:, None] <= mu[None, :] and sigma[:, None] <= sigma[None, :]
    # This creates a 2000x2000 matrix, which is 4M elements, fine.
    
    mu_leq = mu[:, None] <= mu[None, :]  # mu_leq[i, j] = mu[i] <= mu[j]
    sigma_leq = sigma_safe[:, None] <= sigma_safe[None, :]  # sigma_leq[i, j] = sigma[i] <= sigma[j]
    
    # i dominates j if mu[i] <= mu[j] and sigma[i] <= sigma[j] and (mu[i] < mu[j] or sigma[i] < sigma[j])
    # For dominance count of i (how many dominate i), we need j such that mu[j] <= mu[i] and sigma[j] <= sigma[i]
    # and (mu[j] < mu[i] or sigma[j] < sigma[i])
    
    # dominated_by[i] = sum over j of (mu[j] <= mu[i] and sigma[j] <= sigma[i] and (mu[j] < mu[i] or sigma[j] < sigma[i]))
    
    mu_leq_i = mu[None, :] <= mu[:, None]  # mu_leq_i[i, j] = mu[j] <= mu[i]
    sigma_leq_i = sigma_safe[None, :] <= sigma_safe[:, None]  # sigma_leq_i[i, j] = sigma[j] <= sigma[i]
    
    # Strict in at least one
    mu_strict = mu[None, :] < mu[:, None]
    sigma_strict = sigma_safe[None, :] < sigma_safe[:, None]
    
    dominates_i = (mu_leq_i & sigma_leq_i & (mu_strict | sigma_strict))  # dominates_i[i, j] = j dominates i
    
    # Exclude self
    dominates_i = dominates_i * ~np.eye(len(mu), dtype=bool)
    
    dominance_count = np.sum(dominates_i, axis=1)
    
    # Hypervolume-based score:
    # - Lower dominance_count means the point is more non-dominated (better)
    # - Lower distance to ideal means the point is closer to the best (better)
    # - We want high score for good candidates
    
    # Score = exp(-alpha * dist_to_ideal) * exp(-beta * dominance_count / n)
    # This rewards points that are close to the ideal and not dominated by many others.
    
    n = len(mu)
    alpha_hv = 2.0  # Weight for distance to ideal
    beta_hv = 1.0   # Weight for dominance count
    
    hv_score = np.exp(-alpha_hv * dist_to_ideal) * np.exp(-beta_hv * dominance_count / n)
    
    # Also incorporate EI for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1]
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min if ei_max > ei_min else 1.0
    ei_norm = (ei - ei_min) / ei_range
    
    # Combine: 70% hypervolume, 30% EI
    scores = 0.7 * hv_score + 0.3 * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
