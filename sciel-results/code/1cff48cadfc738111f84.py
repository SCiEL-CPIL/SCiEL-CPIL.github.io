import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Local Expected Improvement (LEI) acquisition function.
    Uses a local baseline estimated from the GP posterior (mu, sigma) neighborhood
    to target basin boundaries and local improvements, rather than relying solely
    on the global f_best.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / (cost_safe + 1e-9)
    
    # Ensure inputs are valid
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    n = len(mu)
    
    # Structural Change: Local Baseline via Nearest Neighbors in (mu, sigma) space
    # We compute pairwise distances between all 2000 points in the 2D (mu, sigma) space.
    # This is O(N^2) which is 4M ops, very fast in NumPy.
    
    # Normalize mu and sigma for distance calculation to prevent scale dominance
    # mu can be large (e.g., -78 to 50), sigma is usually small (0 to 5)
    # We standardize both for the distance metric.
    mu_std = (mu - np.mean(mu)) / (np.std(mu) + 1e-9)
    sig_std = (sigma - np.mean(sigma)) / (np.std(sigma) + 1e-9)
    
    # Compute squared Euclidean distances
    # diff_mu = mu_std[:, None] - mu_std[None, :]
    # diff_sig = sig_std[:, None] - sig_std[None, :]
    # dist_sq = diff_mu**2 + diff_sig**2
    
    # More memory efficient way for N=2000:
    mu_col = mu_std.reshape(-1, 1)
    sig_col = sig_std.reshape(-1, 1)
    
    dist_sq = (mu_col - mu_col.T)**2 + (sig_col - sig_col.T)**2
    
    # Find K nearest neighbors for each point
    # Exclude self by setting diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    
    K = 10  # Number of neighbors
    # Get indices of K smallest distances
    # argsort is O(N*K log N) if we use partition or argpartition, but argsort is fine for N=2000
    # We want the indices of the K smallest values in each row
    k_indices = np.argpartition(dist_sq, K, axis=1)[:, :K]
    
    # Gather the mu values of the neighbors
    # We need to map k_indices back to the original mu array
    # neighbor_mu = mu[k_indices] # This gives shape (N, K)
    # But we need to ensure we are picking the correct mu for each row i and col j
    # rows = np.arange(n)[:, None]
    # neighbor_mu = mu[rows, k_indices]
    
    rows = np.arange(n)[:, None]
    neighbor_mu = mu[rows, k_indices]
    
    # Calculate local baseline as the mean of the neighbor mu values
    # This represents the "local best" or "local typical" value
    local_baseline = np.mean(neighbor_mu, axis=1)
    
    # To be more aggressive in finding improvements, we might use the minimum
    # or a weighted mean. Mean is smoother. Let's stick with mean for stability.
    # However, if the local neighborhood is all bad, the baseline is bad.
    # If the local neighborhood contains the global best, the baseline is good.
    
    # Calculate Local Expected Improvement (LEI)
    # LEI = (local_baseline - mu) * Phi(z) + sigma * phi(z)
    z = (local_baseline - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    LEI = (local_baseline - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    LEI = np.maximum(LEI, 0.0)
    
    # Cost-aware adjustment
    # We penalize high cost.
    # Since we don't know the exact spatial location, we rely on the cost array.
    # A higher cost means the point is likely further from the origin (in scaled space).
    # We divide by cost^alpha.
    cost_exp = 1.0
    cost_penalty = np.power(cost, cost_exp)
    
    score = LEI / cost_penalty
    
    # Add a small exploration bonus for high uncertainty points that are also
    # close to the local baseline (z near 0), as these are the most informative
    # for refining the GP in that region.
    # Actually, LEI already handles this. High sigma with z near 0 gives high LEI.
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Normalize scores to [0, 1] for consistency? 
    # The task says "Return 2000 finite scores (higher = evaluate next)".
    # Normalization is safe and helps with numerical stability in the selection loop.
    score_max = np.max(score)
    score_min = np.min(score)
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.zeros_like(score)
        
    return score
