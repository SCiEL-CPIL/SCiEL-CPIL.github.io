"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    score = ei / cost
    
    # Add a small exploration bonus to encourage diversification
    # This helps avoid premature convergence, especially in the early stages
    # when f_best might be poor and EI is dominated by uncertainty
    
    # Compute a scale factor based on the typical EI magnitude
    positive_ei = ei[ei > 1e-10]
    if len(positive_ei) > 0:
        typical_ei = np.median(positive_ei)
    else:
        typical_ei = 1e-3
    
    # Exploration term: reward high uncertainty relative to typical EI
    # This is a soft exploration bonus that doesn't overwhelm the exploitation signal
    exploration_bonus = 0.05 * sigma / (typical_ei + 1e-9)
    
    # Normalize the exploration bonus to be on the same scale as the cost-aware score
    positive_score = score[score > 1e-10]
    if len(positive_score) > 0:
        typical_score = np.median(positive_score)
    else:
        typical_score = 1e-3
    
    # Combine exploitation and exploration
    score = score + exploration_bonus * typical_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
