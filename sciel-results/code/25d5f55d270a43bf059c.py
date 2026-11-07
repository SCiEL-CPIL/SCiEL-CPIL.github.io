import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Structural change: Revert to Expected Improvement (EI)
    # EI is a probabilistic regret minimization approach, structurally different
    # from the UCB (confidence bound) approach in the current program.
    # It balances exploration and exploitation based on the probability of improvement.
    
    # Deadline guard (though EI is very fast, this satisfies the directive)
    start_time = time.time()
    deadline = start_time + 1.0 # 1 second is plenty for 2000 points
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle numerical stability for large negative z
    # If z is very large negative, cdf(z) ~ 0, pdf(z) ~ 0.
    # If z is very large positive, cdf(z) ~ 1, pdf(z) ~ 0.
    
    scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
