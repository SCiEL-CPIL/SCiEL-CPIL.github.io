import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-10
    safe_cost = np.maximum(cost, eps)
    safe_sigma = np.maximum(sigma, eps)
    
    # Thompson Sampling: Sample from the posterior
    # We use the standard normal distribution
    # To ensure some level of reproducibility or just standard behavior,
    # we use np.random.normal.
    
    # Note: In a real BO loop, this function is called sequentially.
    # Each call should ideally use a fresh random state or a seeded one.
    # Since we don't have access to a global RNG state in the function signature,
    # we rely on the global np.random state.
    
    samples = np.random.normal(mu, safe_sigma)
    
    # Calculate cost-adjusted score
    # We want to maximize the score.
    # The objective is to minimize f(x) / cost(x).
    # So we maximize - f(x) / cost(x).
    
    scores = -samples / safe_cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
