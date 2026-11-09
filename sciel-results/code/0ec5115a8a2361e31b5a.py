"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Ackley 2D minimization.
    
    Uses Cost-Adjusted Thompson Sampling.
    Instead of deterministic scores, we sample from the posterior 
    (approximated by mu + sigma * z) and rank by the sampled value 
    divided by cost. This naturally balances exploration and 
    exploitation and is robust to GP miscalibration.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (float)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    t_start = time.monotonic()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Deadline guard
    if time.monotonic() - t_start > 20:
        scores = -mu / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Thompson Sampling: Sample from posterior
    # z ~ N(0, 1)
    # To reduce variance of the selection, we can use a single sample 
    # or average a few. A single sample is standard TS.
    # We scale the noise slightly to control exploration intensity.
    # Standard TS uses scale 1.0.
    # If the GP is overconfident (low sigma), exploration is low.
    # If the GP is underconfident (high sigma), exploration is high.
    
    # We use a single sample for speed.
    z = np.random.normal(0, 1, n)
    
    # Sampled objective values
    # y_sampled = mu + sigma * z
    # We want to minimize y_sampled / cost
    # So score = - (mu + sigma * z) / cost
    
    sampled_values = mu + sigma * z
    scores = -sampled_values / cost
    
    # Sanitize scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check
    if time.monotonic() - t_start > 28:
        return scores
    
    # Optional: Add a small deterministic exploitation bias to prevent 
    # pure exploration if the GP is very uncertain everywhere.
    # This is a hybrid: 0.9 * TS + 0.1 * Greedy (Mean)
    # Greedy score: -mu / cost
    greedy_scores = -mu / cost
    greedy_scores = np.nan_to_num(greedy_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize both to [0, 1] for stable mixing?
    # Or just mix them directly?
    # Direct mixing might be unstable if scales differ.
    # Let's stick to pure TS first as it's the structural change.
    # If TS is too noisy, we can add the bias.
    
    # For now, return pure TS scores.
    return scores
