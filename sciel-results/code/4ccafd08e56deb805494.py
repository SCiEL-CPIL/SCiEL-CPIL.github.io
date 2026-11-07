import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using UCB with adaptive diversity weighting.
    
    Structural change: Instead of look-ahead EI simulation, uses:
    1. Base EI for exploitation
    2. UCB exploration bonus scaled by sigma
    3. Diversity penalty based on mu-value clustering (proxy for spatial density)
    
    This forces exploration of uncertain regions while penalizing candidates
    in densely-populated mu-value regions, encouraging basin-hopping behavior
    appropriate for the 20-basin Styblinski-Tang function.
    """
    N = len(mu)
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute base EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute diversity penalty using mu-value clustering
    # Candidates with mu values similar to many others get penalized
    # Use a Gaussian kernel to estimate local density in mu-space
    
    # Normalize mu values to [0, 1] for kernel bandwidth
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-12:
        mu_norm = np.zeros(N)
    else:
        mu_norm = (mu - mu_min) / mu_range
    
    # Adaptive bandwidth: use a fraction of the mu range
    # Smaller bandwidth = more sensitive to local density
    bandwidth = 0.05 * mu_range  # 5% of the range
    
    if bandwidth < 1e-12:
        # All mu values are nearly identical, no density penalty
        density_penalty = np.zeros(N)
    else:
        # Compute density using a vectorized Gaussian kernel
        # For efficiency, use a coarse grid approximation
        n_grid = 200
        grid = np.linspace(mu_min - bandwidth, mu_max + bandwidth, n_grid)
        
        # Compute density at grid points
        diff_grid = grid[None, :] - mu_norm[:, None] * mu_range
        kernel_vals = np.exp(-0.5 * (diff_grid / bandwidth) ** 2)
        density_grid = np.sum(kernel_vals, axis=0) / (N * bandwidth * np.sqrt(2 * np.pi))
        
        # Interpolate density back to original mu values
        # Use linear interpolation
        from scipy.interpolate import interp1d
        try:
            interp_func = interp1d(grid, density_grid, kind='linear', 
                                   bounds_error=False, fill_value=(density_grid[0], density_grid[-1]))
            mu_density = interp_func(mu)
        except Exception:
            mu_density = np.full(N, np.mean(density_grid))
        
        # Normalize density to [0, 1]
        d_min = np.min(mu_density)
        d_max = np.max(mu_density)
        d_range = d_max - d_min
        if d_range < 1e-12:
            density_penalty = np.zeros(N)
        else:
            density_penalty = (mu_density - d_min) / d_range
    
    # Check time
    if time.time() > deadline - 10.0:
        # Return simple EI if running out of time
        scores = ei.copy()
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # UCB exploration bonus
    # beta controls exploration-exploitation tradeoff
    # Use adaptive beta: higher when sigma is large relative to EI
    sigma_mean = np.mean(sigma)
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    
    # Beta scales with typical uncertainty
    beta = 2.0 * sigma_mean / (ei_max + 1e-12)
    beta = np.clip(beta, 0.5, 5.0)  # Clip to reasonable range
    
    ucb_bonus = beta * sigma
    
    # Diversity-weighted score
    # Penalize high-density regions, boost low-density + high-uncertainty regions
    # The penalty is proportional to density and inversely proportional to sigma
    # (uncertain regions in dense areas are still penalized less)
    diversity_weight = 0.3  # Weight of diversity penalty
    penalty = diversity_weight * density_penalty * (ei_max + 1e-12)
    
    # Final score: EI + UCB bonus - diversity penalty
    scores = ei + ucb_bonus - penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
