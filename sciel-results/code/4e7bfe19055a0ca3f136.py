"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) acquisition function
    # We want to minimize the objective function.
    # The LCB is mu - k * sigma.
    # We return the negative of the LCB so that higher scores correspond to lower LCBs (better candidates).
    # Score = -(mu - k * sigma) = k * sigma - mu
    k = 3.0
    scores = k * sigma - mu
    
    # Ensure finiteness, though GP outputs should be finite
    # If any non-finite values exist, replace them with a very low score to avoid selection
    # But the task says "anything else makes the whole result invalid", implying we should just return valid numbers.
    # Given the GP setup, inputs should be clean.
    
    return scores
# EVOLVE-BLOCK-END
