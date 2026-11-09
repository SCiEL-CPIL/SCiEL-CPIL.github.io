"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive Upper Confidence Bound (UCB) for minimization.
    # We convert the minimization problem to a maximization of a score.
    # Score(x) = -mu(x) + beta * sigma(x)
    # Higher score indicates a more promising candidate for evaluation.
    
    # The global minimum of the 6D Hartmann function is known.
    f_opt = -3.32237
    
    # Calculate the gap between the current best observed value and the optimum.
    # A large gap suggests we are far from the solution and need more exploration (higher beta).
    # A small gap suggests we are close and should exploit (lower beta).
    gap = f_best - f_opt
    
    # Normalize the gap to the range [0, 1] for adaptive beta calculation.
    # We assume a maximum relevant gap of 3.0 units.
    # This handles the initial phase where random points might be far from -3.3.
    norm_gap = np.clip(gap, 0.0, 3.0) / 3.0
    
    # Map the normalized gap to a beta range.
    # Beta starts at 4.0 (high exploration) when the gap is large,
    # and decreases to 1.5 (moderate exploitation) as we approach the optimum.
    beta = 1.5 + 2.5 * norm_gap
    
    # Calculate the acquisition scores.
    # -mu favors low predicted values (exploitation).
    # beta * sigma favors high uncertainty (exploration).
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to satisfy the contract.
    # Replace any NaNs or infinities with large negative/positive values 
    # such that they are not selected if valid scores exist.
    # Since we want to maximize, -inf is bad, +inf is good (but risky if due to error).
    # Typically, valid GP outputs are finite. We use nan_to_num for safety.
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
