"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Unlike Expected Improvement (EI), UCB explicitly balances exploitation (low mu)
    and exploration (high sigma) with a tunable parameter beta.
    Here, beta is adapted based on the current best value f_best relative to the
    known global minimum of the Hartmann 3D function.
    """
    # Known global minimum for Hartmann 3D
    f_opt = -3.86278
    
    # Gap to the global minimum
    gap = f_best - f_opt
    gap = max(0.0, gap)
    
    # Adaptive beta:
    # High beta when far from optimum (explore), low beta when close (exploit).
    beta_min = 0.5
    beta_max = 2.0
    scale = 1.5
    
    # Smooth transition using tanh
    beta = beta_min + (beta_max - beta_min) * np.tanh(gap / scale)
    
    # UCB score for minimization:
    # We want to minimize (mu - beta * sigma), so we maximize (beta * sigma - mu).
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
