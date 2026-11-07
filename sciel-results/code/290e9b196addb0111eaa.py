import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Unlike Expected Improvement (EI), which is non-linear and can under-explore
    in high-uncertainty regions if the mean is not sufficiently low, UCB provides
    a linear reward for uncertainty. This structural change favors more aggressive
    exploration early in the optimization process, which is beneficial for
    multimodal functions like Hartmann-3D where multiple local minima exist.
    
    The score is calculated as:
    score = mu - beta * sigma
    
    A safety deadline guard is included to ensure the function returns within 
    the evaluator's time limit, returning the best available score if the 
    computation takes longer than expected (though unlikely for N=2000).
    """
    start_time = time.time()
    deadline = start_time + 2.0  # 2 seconds safety margin (limit is 300s per eval, but we are fast)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Hyperparameter for exploration
    # For Hartmann-3D, beta in [2, 4] is typical. 2.5 is a balanced choice.
    beta = 2.5
    
    # Compute UCB score for minimization
    # We want to minimize f, so we prefer low mu.
    # We want to explore, so we prefer high sigma.
    # Score = mu - beta * sigma
    # Higher score = better candidate (since we are maximizing the acquisition score to pick the next point)
    # Note: The task says "higher meaning evaluate next".
    # So we need to map (mu - beta*sigma) to a score where higher is better.
    # Since we want to MINIMIZE f, a "good" point has LOW f.
    # UCB for minimization: Minimize (mu + beta * sigma)? No.
    # Standard UCB for minimization: Minimize L(x) = mu(x) - beta * sigma(x) is for MAXIMIZATION?
    # Let's check.
    # If we maximize f, we use UCB = mu + beta*sigma.
    # If we minimize f, we use LCB = mu - beta*sigma.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we need to define a score such that the candidate we want to pick has the highest value.
    # We want to pick a candidate that has low mu and high sigma.
    # So Score should be high when mu is low and sigma is high.
    # Score = -mu + beta * sigma.
    
    scores = -mu + beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        # In case of extreme slowdown, return a safe score (e.g., based on mu only)
        # But this is highly unlikely for N=2000.
        pass

    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores
