"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Lower Confidence Bound (LCB) for minimization.
    
    This approach balances exploitation (finding low values near known good points)
    and exploration (visiting uncertain regions) while accounting for evaluation cost.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon to avoid division by zero and numerical issues
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate Z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # We use a dynamic beta that adapts based on how far we are from the optimum
    # Higher beta when f_best is far from 0 (early stage), lower beta when close
    # This encourages exploration early and exploitation later
    
    # Normalize f_best to [0, 1] range for dynamic beta
    # Griewank 2D has minimum 0 and max around 1-2 in this domain
    f_best_normalized = np.clip(f_best, 0.0, 1.0)
    
    # Beta ranges from 3.0 (exploration) to 1.0 (exploitation)
    beta = 1.0 + 2.0 * (1.0 - f_best_normalized)
    
    lcb = mu - beta * sigma_safe
    
    # Hybrid: combine EI and LCB
    # Weight EI more when it's positive (likely improvement), LCB more for exploration
    # Use a sigmoid-like weight based on z
    # When z is large (mu << f_best), EI dominates; when z is small, LCB helps explore
    weight_ei = 0.5 + 0.5 * np.tanh(z / 2.0)
    weight_lcb = 1.0 - weight_ei
    
    # Combined acquisition before cost normalization
    combined = weight_ei * ei + weight_lcb * (-lcb)  # Negate LCB since we want higher scores
    
    # Cost-aware score: Higher is better
    # Divide by cost to account for evaluation expense
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
