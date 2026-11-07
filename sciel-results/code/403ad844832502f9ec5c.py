import numpy as np
import time

# Hartmann 6D function (kept for completeness/verification, though not used in acquisition)
def _hartmann_6d(x):
    """Compute Hartmann 6D objective value (lower is better)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    val = np.zeros(len(x))
    for i in range(len(x)):
        s = 0.0
        for j in range(4):
            s += alpha[j] * np.sum(A[j] * (x[i] - P[j]) ** 2 * np.cos(np.pi * A[j] * (x[i] - P[j])))
        val[i] = -np.exp(-s)
    return val

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses UCB (specifically LCB for minimization) with linear cost penalty.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle NaNs and Infs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # UCB/LCB Acquisition for Minimization
    # We want to minimize f(x).
    # The Lower Confidence Bound is LCB(x) = mu(x) - beta * sigma(x).
    # We want to select the point with the lowest LCB.
    # Since the evaluator wants "higher = evaluate next", we negate the LCB.
    # Score = -LCB(x) / cost(x) = -(mu(x) - beta * sigma(x)) / cost(x)
    # Score = (-mu(x) + beta * sigma(x)) / cost(x)
    
    beta = 2.0
    
    # Calculate unnormalized score
    # Note: mu is in objective units (e.g., -3.0). 
    # If mu is -3.0, -mu is 3.0.
    # If sigma is 0.5, beta*sigma is 1.0.
    # Numerator = 3.0 + 1.0 = 4.0.
    # If cost is 1.5, Score = 4.0 / 1.5 = 2.66.
    
    # If mu is -1.0 (bad), -mu is 1.0.
    # If sigma is 0.5, Numerator = 1.0 + 1.0 = 2.0.
    # Score = 2.0 / 1.5 = 1.33.
    # The good point has a higher score. Correct.
    
    # If sigma is very high (uncertainty), the score increases, encouraging exploration. Correct.
    
    numerator = -mu + beta * sigma
    score = numerator / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
