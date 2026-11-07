"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Deterministic acquisition function combining Expected Improvement (EI) and
    an uncertainty bonus (sigma) to balance exploitation and exploration.
    
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best (lowest) observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher is better
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Scale EI to be roughly comparable to sigma
    # f_best is typically small (close to 0 for Weierstrass), so EI is in similar units
    # We use a weighted combination: primarily EI, with a small exploration bonus
    # This helps escape local minima by occasionally exploring high-uncertainty regions
    
    # The Weierstrass function has many local minima; pure EI can get stuck.
    # Adding a term proportional to sigma encourages exploration.
    # We use a dynamic coefficient that decreases as we get closer to the optimum.
    
    # Simple approach: weighted sum of EI and sigma
    # EI is already zero when mu >> f_best (explored region) and positive when mu < f_best
    # sigma is high in unexplored regions
    
    # Use a fixed small weight for the exploration term
    # This ensures we don't over-explore but still have some exploration pressure
    weights_ei = 1.0
    weights_sigma = 0.5  # Tuning parameter: balance between exploitation and exploration
    
    scores = weights_ei * ei + weights_sigma * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
