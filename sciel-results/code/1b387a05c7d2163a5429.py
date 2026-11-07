"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return deterministic scores based on sigma to avoid randomness overhead
        # and ensure we return something valid quickly
        return sigma.copy()

    # Strategy: Uncertainty-driven exploration
    # Instead of EI which is greedy, we use a score that heavily weights sigma.
    # This mimics Thompson Sampling's exploratory nature without the stochastic
    # selection overhead, allowing the GP to fill in large unknown regions
    # which is critical for Branin's multiple global minima.
    
    # Normalize sigma to have zero mean and unit variance for consistent scaling
    # across different GP states.
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    if sigma_std < 1e-9:
        # If sigma is constant, fall back to a simple inverse-distance-like proxy
        # or just return sigma (which is constant) + small noise? 
        # But we must return finite scores.
        # If sigma is constant, exploration is uniform.
        return np.ones_like(sigma)
    
    sigma_norm = (sigma - sigma_mean) / sigma_std
    
    # We want high scores for high sigma.
    # We can add a small penalty for high mu (known high values) to avoid
    # revisiting obviously bad areas, but primarily drive by sigma.
    # A linear combination: score = alpha * sigma_norm + beta * (-mu_norm)
    # However, mu and sigma are in different units/scales.
    # Let's stick to a pure exploration strategy first: Score = sigma_norm.
    # This ensures we pick the point of maximum uncertainty.
    
    # To prevent numerical issues or extreme outliers in the score distribution
    # which might not affect argmax but could theoretically cause issues,
    # we clip the normalized sigma.
    scores = np.clip(sigma_norm, -10, 10)
    
    # Ensure finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
