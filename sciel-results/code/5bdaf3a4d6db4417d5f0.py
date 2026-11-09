import numpy as np
import time

def acquisition(mu, sigma, f_best):
       start_time = time.time()
       # Safety margin: 300s total, we are one call.
       # We have plenty of time for 2000 points.
       
       sigma = np.maximum(sigma, 1e-8)
       
       # Monte Carlo EI
       # Draw 50 samples
       # samples shape: (50, 2000)
       # mu shape: (2000,) -> broadcast to (1, 2000)
       # sigma shape: (2000,) -> broadcast to (1, 2000)
       
       n_samples = 50
       # Use standard normal and scale
       # z = np.random.standard_normal((n_samples, len(mu)))
       # samples = mu[None, :] + sigma[None, :] * z
       
       # Optimization: 
       # Instead of generating full samples, we can generate Z and compute
       # improvement = max(0, f_best - (mu + sigma * Z))
       # This is efficient.
       
       z = np.random.standard_normal((n_samples, mu.shape[0]))
       f_samples = mu[None, :] + sigma[None, :] * z
       
       improvements = np.maximum(0, f_best - f_samples)
       mc_ei = np.mean(improvements, axis=0)
       
       # Add small exploration bonus to encourage diversity (Hybrid approach)
       # The pure MC-EI is deterministic in expectation.
       # Adding sigma helps exploration.
       # Let's try a small coefficient, e.g., 0.1
       exploration_bonus = 0.1 * sigma
       
       return mc_ei + exploration_bonus
