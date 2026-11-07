import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (CEI) acquisition.
    
    Instead of EI - penalty, we use EI / cost, which directly optimizes
    the expected improvement per unit of evaluation cost. This is a
    structurally different criterion that naturally balances exploitation,
    exploration, and cost efficiency.
    
    We also add a small exploration bonus proportional to sigma to ensure
    the function doesn't become too greedy in early stages.
    """
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Compute z-score for Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Core: Cost-Effective EI = EI / cost
    # This directly maximizes improvement per unit cost
    cei = ei / cost_safe
    
    # Add a small exploration component to avoid premature convergence
    # The exploration term is also cost-adjusted
    exploration_weight = 0.05
    exploration = exploration_weight * sigma_safe / cost_safe
    
    # Combine CEI with exploration
    scores = cei + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
