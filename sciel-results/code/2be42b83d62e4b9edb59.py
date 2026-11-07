"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # Uncertainty is sigma.
    # Optimistic model: assume the true value is mu - beta * sigma (if we are lucky, it's lower than mean).
    # However, standard UCB for minimization is often formulated as:
    # Select x that minimizes: mu(x) - beta * sigma(x)
    # Equivalently, maximize: -mu(x) + beta * sigma(x)
    
    # Adaptive beta: start with moderate exploration, reduce as uncertainty decreases
    # This helps avoid over-exploration when the model is confident
    mean_sigma = np.mean(sigma)
    if mean_sigma > 1e-10:
        # Normalize beta based on uncertainty level
        # Higher uncertainty -> more exploration
        # Lower uncertainty -> more exploitation
        beta = 1.5 * np.sqrt(np.mean(sigma**2) / (np.mean(sigma**2) + 1e-6))
    else:
        beta = 0.5
    
    scores = -mu + beta * sigma
    
    # Add small random perturbation to break ties and encourage diversity
    # This helps escape local optima in the acquisition landscape
    rng = np.random.default_rng()
    perturbation = rng.normal(0, 0.01, size=scores.shape)
    scores = scores + perturbation
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
