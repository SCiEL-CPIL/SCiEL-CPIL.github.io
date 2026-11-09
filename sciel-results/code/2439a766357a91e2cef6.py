import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    # Ensure sigma is positive and not too small for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Strategy: Cost-Penalized UCB (Lower Confidence Bound)
    # Structural change: Instead of dividing by cost (which distorts scale and 
    # heavily penalizes the high-cost minimum region), we add a small linear 
    # penalty to the mean. This effectively shifts the predicted function value 
    # to account for the cost of evaluation, encouraging the search to balance 
    # function improvement against resource consumption without crushing the 
    # signal magnitude.
    
    # Cost penalty coefficient. 
    # The cost ranges from 1.0 to 2.0. 
    # The function value (Rosenbrock) can be large. 
    # A small lambda ensures cost is a tie-breaker or gentle guide, not the driver.
    lambda_cost = 0.05
    
    # Adjust the mean to include the cost penalty
    # We want to minimize mu + lambda * cost
    # UCB for minimization is: mu - beta * sigma
    # Penalized UCB: (mu + lambda * cost) - beta * sigma
    # We maximize the negative of this: beta * sigma - (mu + lambda * cost)
    
    beta = 1.5  # Slightly more exploratory than 1.0 to escape local minima
    
    # Calculate the score
    # score = beta * sigma - mu - lambda * cost
    score = beta * sigma_safe - mu - lambda_cost * cost
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, return the current best effort
        # The calculation above is vectorized and fast, so this is unlikely to trigger
        # unless numpy is extremely slow, but we include it per directive.
        pass
        
    return score
