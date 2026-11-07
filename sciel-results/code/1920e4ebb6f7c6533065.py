import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Dynamic Thompson Sampling acquisition function.
    
    Strategy:
    Instead of a deterministic score like EI or UCB, we treat the acquisition
    process as sampling from the posterior. For each candidate x, we draw a
    sample y_x ~ N(mu_x, sigma_x). We then return a score that favors the
    candidate with the lowest sampled value.
    
    To ensure the return type is a deterministic score array (as required by the
    interface 'Return 2000 finite scores'), we simulate a single round of
    Thompson Sampling:
    1. Draw one sample per candidate: y_i = mu_i + sigma_i * z_i, where z_i ~ N(0,1).
    2. The 'score' for candidate i is defined such that the candidate with the
       minimum y_i gets the highest score.
       
    However, the interface expects us to return scores for ALL candidates, and the
    evaluator picks the argmax. If we just return -y_i, the argmax will be the
    candidate with the minimum y_i. This effectively performs one step of
    Thompson Sampling.
    
    Why this is different:
    - Current: Deterministic blend of EI and UCB.
    - This: Stochastic selection based on posterior sampling.
    - Benefit: Naturally adapts exploration rate. Early on, sigma is high, so
      samples vary widely, promoting exploration. Later, sigma is low near
      the optimum, so samples cluster around mu, promoting exploitation.
    """
    # Ensure sigma is non-negative to avoid numerical issues in sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random standard normal variables for Thompson Sampling
    # We use a fixed seed internally? No, the evaluator calls this multiple times.
    # We should use the global RNG or generate fresh randomness each time.
    # Using np.random.randn is standard for this approach.
    z = np.random.randn(len(mu))
    
    # Sample from the posterior: y ~ N(mu, sigma)
    y = mu + sigma_safe * z
    
    # We want to minimize y. The evaluator maximizes the returned score.
    # So we return -y. The candidate with the lowest y (best sampled value)
    # will have the highest score (-y).
    
    # Ensure finite values
    scores = -y
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
