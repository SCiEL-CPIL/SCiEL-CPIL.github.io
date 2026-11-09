"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track number of evaluations
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard
    # Not strictly necessary for simple math, but good practice
    
    # Adaptive UCB for minimization
    # Score = -mu + beta * sigma
    
    # Tuned beta:
    # Start with moderate exploration, decay slowly.
    # t is 1..25
    # Try constant beta or slower decay
    # beta = 0.5
    # Or beta = 1.0 / (1.0 + 0.05 * _eval_count)
    
    # Let's try a schedule that keeps beta higher for longer
    beta = 1.0 / (1.0 + 0.05 * _eval_count)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
