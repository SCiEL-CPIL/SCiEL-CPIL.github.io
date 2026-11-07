"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: blend of Expected Improvement and a
    # lower-confidence-bound style term to balance exploitation/exploration.
    #
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu)/sigma
    # LCB: mu - beta * sigma  (we want to minimize, so lower is better;
    #       for a "higher score" acquisition we use -LCB = -mu + beta*sigma)
    #
    # We return a score where higher means "evaluate next".
    # Standard EI already does this: high EI where mu is low and/or sigma is high.
    #
    # To improve over plain EI, we use a weighted combination:
    #   score = w_ei * EI + w_ucb * (-mu + beta * sigma)
    # But since EI already incorporates both, let's try a slightly different
    # formulation: use EI but with a small exploration bonus that is
    # proportional to sigma, which helps when the GP is uncertain.
    #
    # Actually, let's think more carefully. The standard EI for minimization:
    #   EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    #
    # This is well-studied and works well. The key issue is that EI can be
    # too greedy early on. Let's try a modified version that adds a small
    # UCB-like exploration term.
    #
    # Alternative: use a "noisy" EI or a temperature-scaled EI.
    #
    # Let's try: score = EI + gamma * sigma
    # where gamma is a small positive constant. This adds a mild exploration
    # bonus.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma
    # gamma = 0.05 seems reasonable; this makes the acquisition less greedy
    gamma = 0.05
    score = ei + gamma * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
