import numpy as np
from scipy.stats import norm
import time

# ------------------------------- deadline guard ---------------------------------
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 15.0


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _T0)


# ------------------------------- public entry point ------------------------------
def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Cost-Modulated Exploration Bonus.
    
    Structural Change: Instead of a global cost penalty applied to the entire 
    acquisition score (EI + UCB), we apply the cost penalty specifically to the 
    exploration term (sigma). This allows the exploitation signal (EI) to operate 
    at full strength, while explicitly suppressing exploration in expensive regions 
    and enhancing it in cheap regions.
    
    Score = EI + (beta / cost^gamma) * sigma
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)

    # Expected Improvement: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative

    # Cost-Modulated Exploration Bonus
    # beta is the base exploration strength.
    # gamma controls how strongly cost reduces the exploration bonus.
    # cost is in range [1, 2] (1 + mean(x_scaled)), so cost^gamma is in [1, 2^gamma].
    # Dividing by this term reduces the bonus for expensive points.
    beta = 2.0
    gamma = 1.0
    
    # Avoid division by zero or tiny costs
    cost_term = np.power(cost, gamma)
    exploration_bonus = (beta / cost_term) * sigma

    # Combined Score
    score = ei + exploration_bonus

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
