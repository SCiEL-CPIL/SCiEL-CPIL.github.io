"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes the acquisition scores for a given set of candidate points.
    
    This implementation uses the Upper Confidence Bound (UCB) strategy
    with a fixed parameter beta. For minimization, we maximize the optimistic
    lower bound: Score = -mean + beta * std.
    
    A fixed beta of 2.0 provides a good balance between exploration and
    exploitation for small-budget Bayesian optimization on Ackley 2D.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        # Return zeros to ensure finite output and avoid timeout kill mid-calculation
        return np.zeros_like(mu)
    
    # Fixed UCB parameter for exploration
    beta = 2.0
    
    # Calculate UCB scores for minimization
    # We want to find points with potentially low f(x).
    # The optimistic lower bound is mu - beta*sigma.
    # We maximize this "potential" to select the next point.
    # Thus, Score = -(mu - beta*sigma) = -mu + beta*sigma
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
