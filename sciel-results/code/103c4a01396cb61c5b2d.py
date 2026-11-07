import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: track elapsed time to ensure we finish within limits
    start_time = time.time()
    
    # Safe sigma to avoid numerical issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Safe cost to avoid division by zero
    safe_cost = np.maximum(cost, 1e-9)
    
    # UCB (Upper Confidence Bound) for minimization:
    # We want to find points with low function values.
    # The "upper bound" on the function value in a confidence sense is mu + beta*sigma.
    # However, standard UCB for maximization is mu + beta*sigma.
    # For minimization, we are interested in the "lower bound" of the function,
    # which is mu - beta*sigma.
    # The acquisition function should be HIGH for points we want to evaluate.
    # Therefore, we want to MAXIMIZE the potential for low values.
    # Potential low value ~ mu - beta*sigma.
    # So Acquisition = -(mu - beta*sigma) = -mu + beta*sigma.
    # Or equivalently, we can view it as maximizing the "optimism" towards low values.
    
    # Let's use the formulation: Score = (-mu + beta * sigma) / cost
    # beta controls the exploration/exploitation trade-off.
    # A higher beta encourages exploration of uncertain regions (high sigma).
    # The division by cost penalizes expensive evaluations.
    
    # Based on the directive, use a fixed beta (e.g., 2.0) and linear cost penalty.
    beta = 2.0
    
    # Calculate the UCB score for minimization
    # We maximize this score.
    ucb_score = -mu + beta * sigma_safe
    
    # Apply linear cost penalty
    # Dividing by cost means high cost reduces the score.
    # This is equivalent to minimizing (mu - beta*sigma) * cost, 
    # or maximizing (beta*sigma - mu) / cost.
    score = ucb_score / safe_cost
    
    # Ensure all scores are finite
    # If mu is very negative, score is very positive.
    # If sigma is very large, score is very positive.
    # If cost is small (close to 1), score is larger.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # If we're getting close to the deadline, return immediately
    elapsed = time.time() - start_time
    if elapsed > 295:
        return score
    
    return score
