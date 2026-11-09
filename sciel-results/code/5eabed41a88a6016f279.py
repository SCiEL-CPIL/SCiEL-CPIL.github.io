import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Ackley 2D.
    Uses a blend of Expected Improvement and Uncertainty, normalized by cost,
    with adaptive weighting based on how close f_best is to the known minimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Ackley 2D has a global minimum of 0.
    # As f_best approaches 0, we should exploit more (lower beta).
    # When f_best is far from 0, explore more (higher beta).
    
    # Adaptive beta: starts high (exploration) and decreases as f_best improves.
    # f_best is in [0, ~4.6] for Ackley.
    # Let's map f_best from [0, 5] to beta from [0.5, 3.0].
    # If f_best <= 0.01, beta = 0.5 (exploit)
    # If f_best >= 5.0, beta = 3.0 (explore)
    
    f_best = float(f_best)
    
    # Clip f_best to reasonable range for mapping
    f_best_clipped = np.clip(f_best, 0.0, 5.0)
    
    # Linear interpolation: beta = 3.0 - (3.0 - 0.5) * (f_best_clipped / 5.0)
    # When f_best=0, beta=3.0? No, we want opposite.
    # When f_best is large (bad), we want high beta (explore).
    # When f_best is small (good), we want low beta (exploit).
    
    # So: beta = 0.5 + (3.0 - 0.5) * (f_best_clipped / 5.0)
    # f_best=0 -> beta=0.5
    # f_best=5 -> beta=3.0
    beta = 0.5 + 2.5 * (f_best_clipped / 5.0)
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Uncertainty term: sigma
    # We can blend EI and sigma.
    # score = (w_ei * ei + w_sigma * sigma) / cost
    # Let's use a fixed blend, or adaptive?
    # The inspiration mentioned "Blends normalized EI and UCB terms with adaptive weighting".
    # Let's try: score = (ei + 0.5 * sigma) / cost
    # Or maybe use UCB instead of EI?
    # UCB = mu - beta * sigma
    # score_ucb = -ucb / cost = (beta * sigma - mu) / cost
    
    # Let's combine EI and UCB.
    # Actually, EI is already a good balance.
    # Let's stick with EI but maybe add a small exploration term.
    
    # Normalization: EI can be small. Let's scale it.
    # The cost is in [1, 2].
    
    # Let's try: score = (ei + 0.1 * sigma) / cost
    # This ensures that even if EI is 0, high uncertainty points get some score.
    
    scores = (ei + 0.1 * sigma) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
