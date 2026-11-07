"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) strategy for global optimization.
    # Unlike Expected Improvement (EI) which can be overly exploitative,
    # LCB explicitly balances exploitation (low mu) and exploration (high sigma).
    # This is critical for multimodal landscapes like Hartmann 6D with a small budget.
    
    # We want to maximize the returned score.
    # The LCB value is mu - kappa * sigma (to be minimized).
    # So we return the negative of that: -mu + kappa * sigma.
    
    # Kappa is the exploration coefficient.
    # A value around 2.0-4.0 is typical for aggressive exploration.
    kappa = 2.5
    
    scores = -mu + kappa * sigma
    
    # Safety: ensure all scores are finite.
    # If the GP produces non-finite values (unlikely but possible), 
    # we assign a very low score so they are not selected.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
