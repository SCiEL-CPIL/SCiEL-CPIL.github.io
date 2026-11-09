"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a normalized weighted combination
    of Expected Improvement (EI) and Probability of Improvement (PI).
    
    The score is designed to balance exploitation (finding points better than f_best)
    and exploration (reducing uncertainty) while penalizing high-cost evaluations.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases where sigma might be zero or negative (numerical stability)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Normalize EI and PI to have mean 1 (or similar scale) to make them comparable
    # Adding epsilon to avoid division by zero if all improvements are 0
    mean_ei = np.mean(ei) + 1e-9
    mean_pi = np.mean(pi) + 1e-9
    
    ei_norm = ei / mean_ei
    pi_norm = pi / mean_pi
    
    # Normalize Cost to have mean 1
    mean_cost = np.mean(cost) + 1e-9
    cost_norm = cost / mean_cost
    
    # Weighted combination of normalized EI and PI
    # We use equal weights initially, but the normalization makes them comparable.
    # The insight mentioned "adaptive exploitation/exploration weighting".
    # A simple adaptive weight could be based on the average uncertainty.
    # If average sigma is high relative to f_best, we are exploring.
    # However, without history, a fixed 0.5/0.5 blend of normalized metrics is a strong baseline.
    # Let's try a slight bias towards EI as it captures magnitude of improvement, 
    # while PI captures likelihood.
    
    w_ei = 0.6
    w_pi = 0.4
    
    combined_score = w_ei * ei_norm + w_pi * pi_norm
    
    # Final score: Combined improvement potential per unit of normalized cost
    scores = combined_score / cost_norm
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
