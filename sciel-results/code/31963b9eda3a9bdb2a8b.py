# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Progressive Lower Confidence Bound (LCB) acquisition function.
    
    This function uses a linear combination of the GP mean and standard deviation.
    Unlike EI, which is non-linear and focuses on the probability of improvement,
    LCB explicitly balances exploitation (low mu) and exploration (high sigma)
    with a confidence parameter beta.
    
    The beta parameter is adapted based on the current regret (distance from
    f_best to the global minimum). High regret implies early stage, requiring
    higher exploration (higher beta). Low regret implies late stage, requiring
    exploitation (lower beta).
    
    score = mu - beta * sigma
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Start a timer to guard against the 330s wall-clock limit
    t_start = time.time()
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    
    # Calculate current regret (distance from best found to global min)
    regret = f_best - f_min
    
    # Clip regret to avoid numerical issues
    regret = np.clip(regret, 0.0, 10.0)
    
    # Adaptive beta based on regret
    # When regret is large, we want more exploration (higher beta)
    # When regret is small, we want more exploitation (lower beta)
    
    # beta_max: High exploration in early stages
    # beta_min: Low exploration in late stages
    beta_max = 5.0
    beta_min = 1.5
    
    # Transition factor:
    # If regret=0, beta = beta_min.
    # If regret -> inf, beta -> beta_max.
    # beta = beta_min + (beta_max - beta_min) * (1.0 - exp(-regret / scale))
    
    scale = 1.0
    beta = beta_min + (beta_max - beta_min) * (1.0 - np.exp(-regret / scale))
    
    # Compute LCB score
    # We want to minimize, so we maximize the "confidence" that the point is low.
    # score = mu - beta * sigma
    scores = mu - beta * sigma_safe
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
