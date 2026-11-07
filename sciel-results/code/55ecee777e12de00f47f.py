"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates a cost-aware Expected Improvement (EI) score for each candidate.
    
    Strategy:
    1. Use Expected Improvement (EI) for minimization. EI is a deterministic,
       well-calibrated acquisition function that balances exploration and exploitation
       based on the predicted improvement over the current best value (f_best).
       Unlike UCB, EI does not require an arbitrary hyperparameter (kappa) and
       naturally scales with the uncertainty (sigma).
    2. Apply a small, deterministic cost penalty: subtract lambda * cost from the EI score.
       The cost of evaluating a point x is 1 + mean(x). Since the global minimum is
       uniformly distributed in the inner 80% of the domain, a large cost penalty
       would bias the search away from valid minimum locations in the higher-cost
       regions. However, a small penalty (lambda = 0.1) helps conserve the total
       budget (30 units) by preferring cheaper evaluations when the predicted
       improvement is similar, potentially allowing for more total evaluations.
    3. Include a deadline guard to ensure the function returns within the
       330s wall-clock limit, although the computation is very fast.
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
    # We want to minimize f. The improvement of a point x over the current best f_best is:
    # I(x) = max(f_best - f(x), 0)
    # The expected improvement under the GP posterior (mu, sigma) is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma, Phi is the standard normal CDF, and phi is the PDF.
    
    # Compute z
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z) using SciPy for accuracy and speed
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI score
    ei_score = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost penalty:
    # cost is in [1, 2]. A small penalty ensures we don't waste budget on expensive points
    # if the EI is similar.
    lambda_cost = 0.1
    
    # Final score: EI - lambda * cost
    # Higher score means we should evaluate this point next.
    final_score = ei_score - lambda_cost * cost
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END
