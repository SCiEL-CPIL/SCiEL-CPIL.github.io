"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling Strategy:
    Instead of using a deterministic bound like LCB (mu - kappa*sigma), we
    sample a single realization of the function from the Gaussian Process
    posterior. 
    
    Let f_sample = mu + sigma * z, where z ~ N(0, 1).
    The acquisition score for each candidate x_i is f_sample(x_i).
    We select the candidate with the minimum f_sample.
    Since the task requires "higher = evaluate next", we return -f_sample.
    
    This approach is structurally different from LCB/UCB because:
    1. It does not require tuning a hyperparameter kappa.
    2. It provides a stochastic exploration mechanism: regions with high
       uncertainty (high sigma) have a higher variance in f_sample, making
       it more likely to occasionally be the minimum, thus encouraging
       exploration.
    3. It is invariant to the scale of the objective function in a more
       natural way than fixed-kappa LCB.
    
    The cost signal is available but Thompson Sampling inherently handles
    the exploration/exploitation trade-off based on uncertainty. We ignore
    cost for the score calculation itself, as the GP posterior (mu, sigma)
    already encodes the information from previous evaluations. Using cost
    to bias the sample would be a heuristic that might break the optimality
    properties of Thompson Sampling. However, if the cost is a proxy for
    "difficulty" or "distance", and the GP is accurate, the uncertainty
    should already reflect this. 
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Generate a single sample from the posterior
    # z ~ N(0, 1) for all 2000 candidates
    # Using a fixed seed for the RNG inside the function is bad for diversity,
    # but using np.random is fine as it uses the global state.
    # To be safe and reproducible within a single call if needed, we just use np.random.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # f_sample = mu + sigma * z
    f_sample = mu + sigma * z
    
    # The acquisition function should return higher scores for better candidates.
    # Since we are minimizing, lower f_sample is better.
    # So we return -f_sample.
    score = -f_sample
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Return a simple fallback if something went wrong or took too long
         # Fallback: -mu (pure exploitation)
         return -mu
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
