import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # We sample from the posterior distribution at each candidate point
    # The evaluator picks the point with the highest score.
    # Since we want to minimize, we want to pick points where the sampled value is low.
    # However, the evaluator picks the MAXIMUM of the returned scores.
    # So we should return -sampled_value to make low values have high scores.
    
    # Generate a single standard normal random vector of size 2000
    # This is equivalent to sampling f(x) ~ N(mu, sigma) for each x
    # Using a single rng call is efficient.
    # np.random.randn is thread-safe enough for this context? 
    # Actually, using np.random.default_rng() might be better for isolation, 
    # but the global state is fine as long as we don't need reproducibility 
    # across calls in a way that conflicts. 
    # Standard practice in these tasks often allows global numpy random.
    
    # Sample z ~ N(0, 1)
    z = np.random.randn(len(mu))
    
    # Sampled objective value: y = mu + sigma * z
    # We want to minimize y.
    # Evaluator maximizes score.
    # So score = -y = -mu - sigma * z
    
    scores = -mu - sigma * z
    
    # Ensure finite values
    # sigma is standard deviation, should be non-negative.
    # If sigma is 0, score is -mu.
    # If mu is inf or nan, handle it.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
