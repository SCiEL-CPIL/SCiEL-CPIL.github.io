import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Upper Confidence Bound (UCB).
    UCB balances exploitation (low predicted mean) and exploration (high uncertainty).
    The exploration weight beta is adapted based on the current best observed value f_best.
    Higher f_best (early stage) leads to higher beta (more exploration).
    Lower f_best (late stage) leads to lower beta (more exploitation).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is finite and non-negative
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-8)
    
    fb = float(f_best)
    
    # Adaptive beta
    # f_best for sphere 1d on [-5,5] is in [0, 25].
    # We map f_best to a beta range [1.0, 6.0].
    # As f_best decreases, beta decreases.
    beta = 1.0 + fb / 5.0
    
    # Calculate UCB scores
    # We want to maximize this score.
    # Minimizing f corresponds to maximizing -f.
    # UCB = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
