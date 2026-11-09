import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Marginal Thompson Sampling for minimization
    # Sample from the predictive distribution at each candidate point
    # y_i ~ N(mu_i, sigma_i^2)
    
    # Generate random normal samples
    # Use a fixed seed? No, standard TS uses fresh randomness.
    # However, to ensure reproducibility in testing or if the evaluator runs multiple times,
    # we might want to be careful. But typically, BO loops handle the state.
    # We will just use np.random.randn.
    
    z = np.random.randn(len(mu))
    
    # Compute sampled values
    y = mu + sigma * z
    
    # Cost-aware score: We want to minimize y, but also cost.
    # Higher score means "better to evaluate".
    # So we want low y and low cost.
    # Score = -y / cost
    scores = -y / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
