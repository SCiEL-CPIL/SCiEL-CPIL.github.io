"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 330s.
# We track the start time to ensure we return within limits.
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware GP-UCB acquisition function with adaptive exploration.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive Beta for GP-UCB
    # Standard GP-UCB uses beta_t = sqrt(t * log(t^2 d^2 T^2 / pi^2 / 3))
    # Here we don't have t (iteration count) directly, but we can infer progress
    # from f_best and the spread of sigma, or simply use a fixed but well-tuned beta
    # that balances exploration and exploitation.
    # However, to make it "structurally different" and potentially better,
    # we can use a cost-aware variant.
    
    # The core idea of GP-UCB for minimization:
    # We want to maximize: -mu(x) + beta * sigma(x)
    # But we also want to account for cost.
    # A common cost-aware approach is to normalize the gain by the cost.
    # Score = (beta * sigma(x) - mu(x)) / cost(x)
    
    # To improve upon the current LCB (which is essentially this with beta=1.5),
    # we can make beta adaptive. If we are early in the optimization (high uncertainty,
    # f_best is far from minimum), we want higher beta. If we are late (low uncertainty,
    # f_best is close to minimum), we want lower beta.
    
    # Heuristic for "progress":
    # If sigma is generally high, we are exploring. If sigma is low, we are exploiting.
    # Let's use the median sigma as a proxy for the current state of knowledge.
    med_sigma = np.median(sigma)
    
    # If med_sigma is high, increase beta to explore more.
    # If med_sigma is low, decrease beta to exploit more.
    # Baseline beta = 1.5.
    # Let's scale beta based on med_sigma relative to the initial scale of the problem.
    # Since we don't know the exact scale, we can use a simple normalization.
    # Assume the function values are roughly in the range of a few units.
    # If med_sigma > 1.0, we are very uncertain -> beta = 2.5
    # If med_sigma < 0.1, we are very certain -> beta = 0.8
    # Linear interpolation in between.
    
    # To avoid division by zero or extreme values, we clip med_sigma.
    med_sigma_clipped = np.clip(med_sigma, 0.05, 2.0)
    
    # Map med_sigma_clipped from [0.05, 2.0] to beta in [0.8, 2.5]
    # beta = 0.8 + (2.5 - 0.8) * (med_sigma_clipped - 0.05) / (2.0 - 0.05)
    beta = 0.8 + 1.7 * (med_sigma_clipped - 0.05) / 1.95
    
    # Ensure beta is finite and positive
    if not np.isfinite(beta) or beta <= 0:
        beta = 1.5
        
    # Calculate UCB score
    # Higher score means better (lower predicted minimum, high uncertainty) and cheaper.
    # ucb_value = mu - beta * sigma  (lower is better for the function value)
    # acquisition_score = -ucb_value / cost = (beta * sigma - mu) / cost
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (beta * sigma - mu) / cost
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Check deadline
    # If we are extremely close to the limit, we could return a simpler score,
    # but UCB is O(N) and very fast, so this is likely unnecessary.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
