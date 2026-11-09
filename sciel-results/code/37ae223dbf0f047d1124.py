import numpy as np

def acquisition(mu, sigma, f_best):
          sigma_safe = np.maximum(sigma, 1e-9)
          
          # Sample from the posterior
          # y ~ N(mu, sigma^2)
          # We want to minimize y, so we maximize -y
          # Draw a single random number z ~ N(0,1)
          # y_i = mu_i + sigma_i * z
          
          z = np.random.normal(0, 1)
          
          scores = -(mu + sigma_safe * z)
          
          # Ensure finite
          scores = np.where(np.isfinite(scores), scores, -1e10)
          
          return scores
