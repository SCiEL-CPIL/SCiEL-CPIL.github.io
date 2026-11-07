import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive K based on f_best
    # High f_best (early) -> High K (exploration)
    # Low f_best (late) -> Low K (exploitation)
    
    # Clamp f_best to avoid issues if it's negative (shouldn't be for sphere)
    f_best_pos = max(f_best, 0.0)
    
    # Map f_best to K
    # As f_best -> inf, K -> K_max
    # As f_best -> 0, K -> K_min
    K_min = 5
    K_max = 200
    
    # Smooth transition
    # Using a simple rational function
    K = int(K_min + (K_max - K_min) * (f_best_pos / (f_best_pos + 1.0)))
    
    # Ensure K is at least 1
    K = max(K, 1)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate samples
    # Shape: (N_candidates, K_samples)
    samples = np.random.normal(mu[:, np.newaxis], sigma_safe[:, np.newaxis], size=(mu.shape[0], K))
    
    # Min over samples
    min_vals = np.min(samples, axis=1)
    
    # Score is negative of min
    score = -min_vals
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
