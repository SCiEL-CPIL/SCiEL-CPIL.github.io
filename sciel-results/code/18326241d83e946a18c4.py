import numpy as np
import time

def acquisition(mu, sigma, f_best):
       start_time = time.time()
       
       # UCB for minimization:
       # We want to maximize the "optimistic" lower bound?
       # No, we want to find the minimum.
       # The lower bound of the true value is mu - beta * sigma.
       # We want to pick the point with the lowest lower bound.
       # So we minimize (mu - beta * sigma).
       # Equivalently, we maximize (beta * sigma - mu).
       
       beta = 1.5
       scores = beta * sigma - mu
       
       # Handle non-finite values
       scores = np.where(np.isfinite(scores), scores, -np.inf)
       
       # Safety margin
       elapsed = time.time() - start_time
       if elapsed > 280.0:
           return scores
           
       return scores
