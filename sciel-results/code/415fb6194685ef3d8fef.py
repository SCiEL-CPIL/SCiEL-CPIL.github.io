import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior
    # We want to minimize f. 
    # Sample f_hat = mu + sigma * Z, where Z ~ N(0, 1)
    # We want to pick the point with the minimum f_hat.
    # So we return -f_hat as the score (higher is better).
    
    # Use a fixed seed? No, random is fine.
    # However, to be safe against any "deterministic" expectations, 
    # maybe we should stick to deterministic?
    # The prompt doesn't forbid randomness.
    
    z = np.random.randn(len(mu))
    samples = mu + sigma * z
    
    # Return negative samples so that min sample -> max score
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
