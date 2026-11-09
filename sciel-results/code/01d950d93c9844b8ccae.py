"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Dynamic UCB with exploration decay.
    # We assume a typical BO loop length. Since we don't have the step count,
    # we approximate the decay based on the variance of the posterior relative to the function range,
    # or simply use a fixed, well-tuned kappa that balances exploration/exploitation better than EI for multimodal functions.
    # However, a "different strategy" suggests moving away from static EI.
    # A common improvement is the Gaussian Process Upper Confidence Bound (GP-UCB).
    # score = mu + kappa * sigma
    
    # For Weierstrass, which is highly multimodal, a high kappa is needed initially.
    # Since we cannot track steps easily without state, and the function is stateless,
    # we can use a heuristic that adapts to the current uncertainty.
    # Alternatively, we can use a fixed kappa that is known to perform well on such benchmarks.
    # EI (kappa=0 effectively) often gets stuck. UCB with kappa ~ 2-5 is often better for multimodal.
    
    # Let's try a moderate UCB.
    kappa = 2.5
    scores = mu + kappa * sigma
    
    # Ensure finite scores
    return scores
# EVOLVE-BLOCK-END
