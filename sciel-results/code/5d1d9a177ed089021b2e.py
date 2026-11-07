import numpy as np
import time

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    Combines Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with cost-awareness.
    
    1. Computes EI for minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma.
    2. Computes LCB for minimization: LCB = mu - kappa * sigma.
       We want to maximize -LCB, so score_lcb = -mu + kappa * sigma.
    3. Normalizes both EI and score_lcb to [0, 1] range.
    4. Blends them: score = w_ei * EI_norm + w_lcb * LCB_norm.
    5. Divides by cost to make it cost-aware (lower cost -> higher score).
    6. Returns the final scores.
    
    This approach balances exploitation (EI) and exploration (LCB) while
    accounting for the evaluation cost.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma
    
    # Use standard normal CDF and PDF
    # scipy.stats.norm.cdf and norm.pdf are efficient
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Compute LCB score (we want to maximize -LCB = -mu + kappa * sigma)
    # Use a moderate exploration constant
    kappa = 1.5
    lcb_score = -mu + kappa * sigma
    
    # Normalize EI and LCB score to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(arr)
    
    ei_norm = normalize(ei)
    lcb_norm = normalize(lcb_score)
    
    # Blend EI and LCB
    # Weights: EI is more exploitative, LCB is more exploratory
    # Let's start with a 50/50 blend
    w_ei = 0.5
    w_lcb = 0.5
    
    blended_score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware adjustment: divide by cost
    # This penalizes expensive evaluations
    scores = blended_score / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
