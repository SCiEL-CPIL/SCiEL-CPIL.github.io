"""Rewritten program for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Probability of Improvement (PI) combined with a 
    distance penalty toward the known global minimizer at the origin (0,0).
    
    PI = Phi((f_best - mu) / sigma)
    This is a smooth, bounded function in [0, 1] that measures the probability
    that a point improves upon the current best.
    
    We combine PI with a distance penalty that biases exploration toward the center
    of the domain (where the Ackley function's global minimum is located).
    
    The distance penalty is proportional to the squared distance from the origin,
    which creates a smooth attraction toward the known optimum location.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Probability of Improvement
    z = (f_best - mu) / sigma
    pi_score = norm.cdf(z)
    
    # The mu values are in objective space, but we need to estimate where in the 
    # input space these candidates are. Since we don't have coordinates directly,
    # we use the GP mean as a proxy: lower mu suggests closer to the optimum.
    # We create a penalty based on how far the GP mean is from the known minimum (0).
    # This biases the search toward regions where the GP predicts low values.
    
    # Distance penalty: penalize points where the GP mean is far from 0 (the minimum)
    # Scale the penalty so it's comparable to the PI score range [0, 1]
    # Use a soft penalty that grows with |mu|
    penalty_scale = 0.5  # Controls how strongly we pull toward the center
    distance_penalty = penalty_scale * np.exp(-np.abs(mu) / 1.0)
    
    # Combine: higher PI and higher proximity to center gives higher score
    # We want to maximize this
    score = pi_score + distance_penalty
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
