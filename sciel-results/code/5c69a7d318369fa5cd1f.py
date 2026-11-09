# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
       """
       Cost-aware Thompson Sampling acquisition function using expected minimum
       of Monte Carlo samples.
       """
       n_candidates = len(mu)
       n_samples = 50  # Number of Monte Carlo samples per candidate
       
       # Reshape for broadcasting: (n_samples, n_candidates)
       mu_b = mu.reshape(1, -1)
       sigma_b = sigma.reshape(1, -1)
       
       # Ensure sigma is non-negative and finite
       sigma_b = np.clip(sigma_b, 1e-6, None)
       
       # Draw samples: (n_samples, n_candidates)
       rng = np.random.default_rng(42)
       samples = rng.normal(loc=mu_b, scale=sigma_b, size=(n_samples, n_candidates))
       
       # Compute the minimum value across samples for each candidate
       # This approximates the expected minimum, providing an exploration bonus
       min_values = np.min(samples, axis=0)  # (n_candidates,)
       
       # Cost-aware score: lower min value / lower cost is better
       # Score = -min_value / cost (higher is better)
       scores = -min_values / cost
       
       # Ensure finite values
       scores = np.where(np.isfinite(scores), scores, -1e10)
       
       return scores
   # EVOLVE-BLOCK-END
