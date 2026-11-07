"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates a hybrid acquisition score for Bayesian Optimization.
    
    Strategy:
    1. Combine Expected Improvement (EI) and Upper Confidence Bound (UCB).
       - EI is strong for exploitation when the GP has learned the landscape.
       - UCB is strong for exploration in early stages or when uncertainty is high.
       - The inspirations show that pure UCB (kappa=1.0) scored ~1.48, while
         pure EI with cost penalty scored ~1.00. This suggests that exploration
         is more valuable than strict cost-aware exploitation in this budget-constrained
         setting, or that the cost penalty in EI was harmful.
       
    2. The hybrid score is: score = EI + alpha * UCB_component
       where UCB_component = kappa * sigma (the exploration term of UCB).
       We use a small alpha to keep EI as the primary driver but allow sigma to
       boost candidates with high uncertainty, similar to how UCB works.
       
    3. We do NOT apply a cost penalty. The inspirations showed that penalizing
       cost biases the search away from regions where the minimum might be.
       The budget constraint is handled by the loop itself, not by the acquisition
       function's bias.
       
    4. We scale the UCB component to be comparable to EI. Since EI is in objective
       units and sigma is in objective units, we can add them directly.
       We use a small weight for the sigma term to avoid over-exploration.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z) using SciPy for accuracy and speed
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei_score = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # UCB exploration component: kappa * sigma
    # We use a small kappa to add exploration without dominating EI.
    # The pure UCB inspiration used kappa=1.0 and scored well.
    # Here we use a smaller value to blend with EI.
    kappa = 0.5
    
    ucb_component = kappa * sigma_safe
    
    # Hybrid score: EI + UCB_component
    # This balances exploitation (EI) and exploration (sigma term)
    final_score = ei_score + ucb_component
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END
