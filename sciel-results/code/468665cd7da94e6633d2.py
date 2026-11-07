"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _START_TIME > TIME_LIMIT:
        # Return a neutral score or the best available (mu) to avoid crashing
        # In a tight loop, we might just return mu to stop exploring
        return mu
    
    # Upper Confidence Bound (UCB)
    # kappa controls exploration. Higher kappa = more exploration.
    # For Michalewicz with low noise, exploration is key to escape local minima.
    kappa = 5.0
    return mu - kappa * sigma
# EVOLVE-BLOCK-END
