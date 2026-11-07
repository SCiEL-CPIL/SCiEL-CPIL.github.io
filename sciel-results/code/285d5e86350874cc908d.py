"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower bound of Expected Improvement (LBEI) with a small exploration parameter.
    # This tends to balance exploitation and exploration more aggressively than vanilla EI.
    # We use a small epsilon to ensure we don't completely ignore exploration.
    eps = 1e-3
    z = (f_best - mu) / (sigma + 1e-9)
    # LBEI formula: (f_best - mu) * Phi(z) + sigma * phi(z) is EI.
    # LBEI is typically: (f_best - mu) * Phi(z) + sigma * phi(z) - eps * something?
    # Actually, a common variant is just EI but with a modified threshold or using
    # Probability of Improvement (PI) blended with EI.
    
    # Let's try a blend of Expected Improvement and Probability of Improvement.
    # PI = Phi(z)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # A simple and effective alternative is the "Expected Regret" or just using
    # a weighted sum. However, a very strong baseline that often outperforms EI
    # in 1D is the "Upper Confidence Bound" (UCB) in the minimization context,
    # which is equivalent to maximizing: -mu + kappa * sigma.
    # Since we want to return scores where higher is better, and we are minimizing f,
    # we want to pick points with low mu and high sigma.
    # UCB score = -mu + kappa * sigma.
    
    # Let's try UCB with a tuned kappa.
    # A common heuristic for kappa is sqrt(2 * log(t)) but we don't have t.
    # In 1D, a fixed kappa around 1.0 to 2.0 often works well.
    # Let's try kappa = 1.5.
    
    kappa = 1.5
    # To avoid issues with very large sigma, we can clip or just use it directly.
    # The problem states sigma is a standard deviation in objective units.
    
    scores = -mu + kappa * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
