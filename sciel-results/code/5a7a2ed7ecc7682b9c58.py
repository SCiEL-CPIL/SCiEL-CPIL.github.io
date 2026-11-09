import numpy as np

def acquisition(mu, sigma, f_best):
        # Thompson Sampling acquisition function
        # Draw a single sample from the posterior predictive distribution
        # f(x) ~ N(mu(x), sigma(x)^2)
        
        # Ensure sigma is non-negative to avoid issues with sqrt or distribution logic
        sigma = np.asarray(sigma)
        mu = np.asarray(mu)
        
        # Clip sigma to a small positive value to avoid zero variance issues if any
        sigma = np.maximum(sigma, 1e-9)
        
        # Generate standard normal noise
        noise = np.random.standard_normal(mu.shape)
        
        # Sample function values
        # f_sample = mu + sigma * noise
        f_sample = mu + sigma * noise
        
        # We want to minimize the function value.
        # The evaluator selects the candidate with the highest acquisition score.
        # Therefore, we return the negative of the sampled function value.
        scores = -f_sample
        
        # Ensure all scores are finite
        # If numerical issues cause inf/nan, replace with a very low score
        scores = np.where(np.isfinite(scores), scores, -1e10)
        
        return scores
