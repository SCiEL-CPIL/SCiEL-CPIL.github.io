import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 325.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)

    # Sanitize
    sigma = np.maximum(sigma, 1e-9)
    cost_arr = np.maximum(cost_arr, 1e-9)

    # f_best is a scalar
    # cost_arr is an array, but 'cost' in the prompt signature is the array of costs for the 2000 candidates?
    # "mu, sigma and cost have shape (2000,)."
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30..."
    # The 'cost' argument passed to acquisition is the cost of evaluating *each* of the 2000 candidates.
    # It is NOT the total budget spent. It is the per-point cost.
    # Wait, the prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # And "mu, sigma and cost have shape (2000,)."
    # So `cost` is the array of costs for the 2000 candidate points.
    # We do NOT know the total budget spent in this function call directly, 
    # unless we infer it from f_best? No.
    # However, the loop runs "While total cost is below 30".
    # We don't have access to the total cost history.
    # So we cannot use total budget for scheduling.
    
    # Therefore, we must rely on f_best and the distribution of costs/sigmas.
    
    # Let's stick to f_best for scheduling, as in Inspiration 1.
    
    # Structural Change: 
    # Instead of linear kappa, use a kappa that is also sensitive to the *relative* uncertainty.
    # But TS already uses sigma.
    
    # Let's refine the kappa calculation to be slightly more aggressive in the mid-game.
    # Inspiration 1: kappa = 1 + f_best / 50.
    # If f_best is 10, kappa = 1.2.
    # If f_best is 1, kappa = 1.02.
    
    # Let's try a floor of 1.1 to prevent premature convergence to GP local minima.
    # And a slightly faster decay.
    
    f_best_clamped = max(0.0, float(f_best))
    
    # If f_best is high, explore a lot.
    # If f_best is low, explore a little less, but not zero.
    kappa = 1.1 + f_best_clamped / 40.0
    kappa = np.clip(kappa, 1.1, 5.0)
    
    # Sample from posterior
    std = kappa * sigma
    y = np.random.normal(mu, std)
    
    # Cost-aware score
    # Higher score is better.
    # We want low y and low cost.
    scores = -y / cost_arr
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
