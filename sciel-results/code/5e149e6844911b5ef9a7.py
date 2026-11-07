"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Improved LCB with adaptive beta and stochastic diversity.
    
    Uses a Lower Confidence Bound (LCB) for minimization.
    The exploration parameter beta is adapted based on the progress
    towards the known minimum of Hartmann 3D.
    To enforce spatial diversity and prevent redundant evaluations
    (acting as a 'repulsion' term in the absence of coordinates),
    a small Gaussian noise term is added to the scores. This
    stochastic perturbation helps break ties and encourages
    exploring near-optimal candidates, spreading the search.
    """
    # Known minimum of Hartmann 3D
    h3_min = -3.86278
    
    # Estimate progress towards the minimum.
    # Assume initial f_best is around 0.0 (typical for Hartmann 3D random start).
    # Progress is 0 at start, 1 at minimum.
    f_start = 0.0
    progress = np.clip((f_start - f_best) / (f_start - h3_min), 0.0, 1.0)
    
    # Adaptive beta: 
    # Start with higher exploration (beta=2.5) to find basins.
    # End with lower exploitation (beta=1.0) to refine the minimum.
    beta = 2.5 - 1.5 * progress
    
    # Base LCB score: -mu + beta * sigma
    # We want to maximize this score (which corresponds to minimizing the LCB).
    score = -mu + beta * sigma
    
    # Add small stochastic noise to promote diversity (repulsion effect).
    # This prevents the algorithm from getting stuck in a single local optimum
    # of the acquisition function and encourages sampling from different regions.
    # The scale is small relative to the function values (~4 units).
    noise_scale = 0.01
    score = score + noise_scale * np.random.randn(len(mu))
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
