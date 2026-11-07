"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 290.0  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CA-EI)
    with an explicit exploration bonus.
    
    Strategy:
    1. Calculate standard Expected Improvement (EI) relative to f_best.
    2. Divide EI by the evaluation cost to favor cheaper evaluations.
    3. Add a cost-normalized exploration bonus proportional to sigma (uncertainty).
       This helps escape local minima in Rastrigin by encouraging the search to
       visit regions with high predictive variance, which is often where the
       global minimum (or other basins) lies if the current f_best is stuck in
       a local one.
    
    Why this is different:
    - The current program uses Thompson Sampling (stochastic, single sample).
    - This program uses deterministic EI, which is more stable and theoretically
      grounded for Bayesian Optimization.
    - The explicit addition of sigma/cost addresses the "stagnation" signal by
      forcing exploration when uncertainty is high, which TS might not do
      aggressively enough if the sampled values are not low enough.
    - Cost awareness is integrated directly into both the improvement term and
      the exploration term.
    """
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple deterministic score if we are running out of time
        # This shouldn't happen often, but prevents timeout failure
        # Use a simple cost-aware negative mean
        return -mu / cost

    # Ensure sigma is positive to avoid division issues, though GP should output >0
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # 1. Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: We are minimizing. So improvement is f_best - mu.
    
    z = (f_best - mu) / sigma_safe
    # Handle cases where mu > f_best (no improvement) -> z < 0
    # norm.cdf(z) will be small, norm.pdf(z) is symmetric.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-Adjust the EI
    # We want to maximize benefit per cost.
    # cost is (2000,) array, min 1.0
    cost_safe = np.maximum(cost, 1e-6)
    
    cost_adjusted_ei = ei / cost_safe
    
    # 3. Add Exploration Bonus
    # The cell findings suggest that adding an exploration bonus proportional to
    # sigma/cost helps escape local minima.
    # Let's add a term: beta * (sigma / cost)
    # We need to choose beta. 
    # The magnitude of EI can vary. 
    # A common heuristic is to balance exploitation (EI) and exploration (sigma).
    # Let's use a dynamic beta or a fixed one. 
    # Given Rastrigin is multi-modal, exploration is key.
    # Let's try a moderate exploration weight.
    # Note: The units of EI are objective values. Sigma is also objective values.
    # So they are comparable.
    
    # Let's set beta = 0.5. This means we value uncertainty equally to half the
    # expected improvement. This can be tuned, but 0.5 is a reasonable start
    # for a "different strategy" that emphasizes exploration more than pure EI.
    beta = 0.5
    exploration_bonus = beta * (sigma_safe / cost_safe)
    
    score = cost_adjusted_ei + exploration_bonus
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
