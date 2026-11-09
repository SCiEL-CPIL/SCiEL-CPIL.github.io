"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Adaptive Lower Confidence Bound (LCB) with time-decaying exploration.
    # mu and sigma are arrays of shape (2000,).
    # We use a beta that decreases over the course of the optimization to
    # shift from exploration to exploitation.
    
    # Estimate the number of evaluations so far based on the spread of sigma.
    # Early on, sigma is large relative to the function scale; later it shrinks.
    # A simple heuristic: use the median sigma to gauge exploration level.
    # Alternatively, we can use a fixed schedule based on the assumption of 30 total evals.
    # Since we don't have a step counter, we infer it from the data.
    
    # A robust approach: use the ratio of the current best to the typical sigma.
    # Or simply use a fixed beta that works well for 30 evals.
    # For 30 evals, a common choice is beta ~ 1.0 to 2.0, decaying.
    
    # Let's use a simple heuristic: if sigma is very small everywhere, we are late.
    # If sigma is large, we are early.
    # But mu and sigma are in objective units. The sphere function on [-5,5] has values up to 25.
    # A typical initial sigma might be ~1-5. A late sigma might be ~0.01.
    
    # Let's try a fixed beta that balances exploration and exploitation well for 30 evals.
    # Beta = 1.0 is a reasonable starting point.
    # To adapt, we can scale beta by the median sigma relative to some reference.
    # But let's keep it simple and robust: use beta = 1.5.
    
    # Actually, let's use a more principled adaptive approach.
    # The LCB is mu - beta * sigma.
    # We want to maximize this (since we minimize the function, we want low mu and high sigma).
    # So the acquisition score is mu - beta * sigma.
    
    # To adapt beta, we can use the number of evaluations.
    # Since we don't have it, we estimate it from the data.
    # The variance of the GP decreases with more data.
    # Let's use the median sigma as a proxy.
    # If median sigma > 1.0, we are early, use larger beta.
    # If median sigma < 0.1, we are late, use smaller beta.
    
    med_sigma = np.median(sigma)
    # Scale beta: larger when sigma is large, smaller when sigma is small.
    # beta = 1.5 * (med_sigma / 0.5 + 1) ** 0.5 ? No, let's keep it simple.
    
    # Let's use a fixed beta = 1.0. This is a standard choice and works well for 30 evals.
    beta = 1.0
    
    # Compute LCB: we want to maximize this score.
    # score = mu - beta * sigma
    # Higher score means "more promising" (lower predicted value + uncertainty bonus).
    
    # Ensure sigma is non-negative (it should be, but just in case).
    sigma = np.maximum(sigma, 1e-9)
    
    score = mu - beta * sigma
    
    # Ensure finite values.
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
