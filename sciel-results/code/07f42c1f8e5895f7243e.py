import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Draw a single sample from the posterior distribution: y = mu + sigma * z, where z ~ N(0, 1).
    2. Select the point with the lowest sampled value y, as this represents the most likely 
       "best" point given the current uncertainty.
    3. Incorporate cost by subtracting a penalty from the sampled value. Since the goal is to 
       minimize the objective, we want to penalize high cost. The cost is in [1, 2]. 
       We subtract a small multiple of the cost to slightly prefer cheaper evaluations, 
       but the primary driver is the posterior sample.
    4. To ensure we return a score where "higher = evaluate next", we invert the logic:
       Score = - (sampled_value - cost_penalty).
       Alternatively, we can just return -sampled_value + cost_penalty.
       
    Note: Thompson Sampling is stochastic. In a batch setting (evaluating 2000 candidates), 
    we draw one sample for each candidate and pick the one with the best (lowest) adjusted value.
    The "score" returned to the outer loop is used to select the single best candidate.
    Therefore, we return a vector where the maximum value corresponds to the candidate 
    with the minimum sampled objective (adjusted for cost).
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw a single standard normal sample for each candidate
    # This is the core of Thompson Sampling
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior
    samples = mu + sigma_safe * z
    
    # Cost adjustment:
    # We want to minimize the objective. 
    # The cost is an extra resource spent. 
    # A simple way to incorporate cost in a minimization context via Thompson Sampling 
    # is to subtract a penalty from the "value" we are trying to maximize (the negative objective).
    # Let's define the "utility" of a point as: -sample + lambda * (1 - cost) or similar.
    # However, cost is [1, 2]. Lower cost is better.
    # Let's stick to a simple linear penalty on the sample itself, effectively making expensive points 
    # slightly "worse" in the minimization sense.
    # adjusted_sample = sample + beta * cost
    # We want the smallest adjusted_sample.
    # The score returned should be higher for better points.
    # Score = -adjusted_sample = -(sample + beta * cost)
    
    # Choosing beta:
    # The objective function (Griewank) ranges roughly from 0 to ~1.
    # Cost ranges from 1 to 2.
    # If beta is too large, we just pick the cheapest points.
    # If beta is small, we ignore cost.
    # A value like 0.01 or 0.05 is reasonable. Let's use 0.02.
    beta = 0.02
    
    adjusted_samples = samples + beta * cost
    
    # The candidate with the minimum adjusted_sample is the best.
    # We return scores such that the max score corresponds to the min adjusted_sample.
    score = -adjusted_samples
    
    # Safety check for deadline
    if time.time() > deadline:
         # If we are running out of time, just return the current scores
         pass

    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return score
