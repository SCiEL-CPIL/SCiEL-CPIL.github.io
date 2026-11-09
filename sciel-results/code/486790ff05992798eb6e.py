import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Compute Phi(z) and phi(z)
    # Using scipy.stats.norm for accuracy and speed
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Adaptive exploration weight
    # The Beale function has a minimum of 0 and values can range up to ~50-100 in the domain.
    # Early on, f_best is high (e.g., 20-50), we want to explore.
    # Late on, f_best is low (e.g., <1), we want to exploit.
    
    # The current program used beta = 1.5 * sqrt(f_best + 0.1).
    # Inspiration 1 used a blend of EI and UCB with a sigmoid-like beta.
    # Let's try a hybrid: Use EI as the base, and add a small UCB-like exploration term.
    # Or, weight the EI and UCB.
    
    # A common approach is to use EI, which naturally balances exploration and exploitation.
    # However, EI can be too exploitative in the early stages if the GP is not well-calibrated.
    # The current program (score 3.68) used pure LCB (score = -mu + beta*sigma).
    # Inspiration 2 (score 1.0) used pure EI.
    # Inspiration 1 (score 1.76) used EI + beta*sigma.
    
    # Let's try a weighted combination of EI and UCB.
    # UCB score for minimization (higher is better): -mu + beta * sigma
    # We can normalize them or just add them.
    
    # Let's refine the beta from the current program.
    # beta = 1.5 * sqrt(f_best + 0.1)
    # If f_best = 50, beta = 1.5 * 7.07 = 10.6
    # If f_best = 1, beta = 1.5 * 1.05 = 1.57
    # If f_best = 0.1, beta = 1.5 * 0.45 = 0.67
    
    # This seems reasonable. Let's try adding a small fraction of UCB to EI.
    # Or, let's try the "EI + beta * sigma" approach from Inspiration 1, but with the current program's beta.
    
    beta = 1.5 * np.sqrt(f_best + 0.1)
    
    # Score = EI + beta * sigma
    # This adds an explicit exploration bonus on top of EI.
    # EI already rewards high sigma when mu is high, but this strengthens it.
    
    score = ei + beta * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
