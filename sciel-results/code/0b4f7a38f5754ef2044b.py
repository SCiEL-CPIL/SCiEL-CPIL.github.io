"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Stateful storage for the exploration parameter
# Initialized to a value that encourages exploration in early steps
# and decays slightly or stays constant to balance exploitation later.
# Since we don't know the total steps (30 evals, 5 initial -> 25 BO steps),
# we can use a simple counter or a fixed aggressive beta.
# LCB: mu - beta * sigma
# A higher beta means more exploration.
# Standard EI is roughly equivalent to LCB with a beta that depends on the current best.
# Let's try a fixed, relatively high beta to force exploration of the 2D landscape.
# Michalewicz 2D has multiple peaks. EI often gets stuck in the first found peak.
# LCB with beta ~ 1.5 - 2.0 is a common starting point for exploration.

_global_step = 0

def acquisition(mu, sigma, f_best):
    global _global_step
    _global_step += 1
    
    # Adaptive beta: Start high to explore, decrease slowly to exploit.
    # With ~25 steps, we can start at 2.0 and drop to 0.5.
    # Linear decay: beta = 2.0 - (1.5 * _global_step / 25)
    # Or simply a fixed high value if we suspect EI is too greedy.
    # Let's use a fixed beta of 1.5 to start, which is a significant exploration boost over standard EI.
    # Actually, let's make it adaptive to be safe.
    # If _global_step is 1, beta=2.0. If 25, beta=0.5.
    beta = 2.0 - (1.5 * (_global_step - 1) / 25.0)
    
    # Ensure beta doesn't go negative
    if beta < 0.5:
        beta = 0.5
        
    # LCB score: higher is better (since we are minimizing, we want low mu and high sigma)
    # score = mu - beta * sigma
    return mu - beta * sigma
# EVOLVE-BLOCK-END
