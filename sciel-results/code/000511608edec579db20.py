"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def _compute_cost(x):
    """Compute cost for points x of shape (n, 4)."""
    x_scaled = x / 10.0  # scale to [0, 1]
    return 1.0 + np.mean(x_scaled, axis=1)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of computing a scalar improvement metric like EI, this strategy
    samples from the posterior distribution of the objective function.
    We want to minimize the function, so we select the candidate with the
    lowest sampled value. To make the output compatible with the "higher is better"
    interface, we return the negative of the sampled value.
    
    We also incorporate a cost-awareness factor. Since evaluating expensive points
    yields less "value per unit cost", we can bias the sampling or the final score.
    However, pure Thompson Sampling is known to be robust. A common cost-aware
    extension is to penalize the sampled value by the cost, or to adjust the
    variance. Here, we use a simple cost-normalized score:
    score = -sampled_value / cost
    
    This encourages finding low values that are cheap to obtain.
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clamp sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Draw a sample from N(mu, sigma^2)
    # We use a fixed random seed for reproducibility within the run? 
    # No, the evaluator runs multiple instances. 
    # However, the acquisition function is called multiple times in the loop.
    # Using a global RNG state is fine, as it provides stochasticity.
    
    # Generate random normal numbers
    z = np.random.randn(len(mu))
    samples = mu + sigma * z
    
    # We want to minimize the function. 
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return the negative of the sampled value.
    # To be cost-aware, we divide by cost.
    # score = -sample / cost
    
    # Safety: ensure no division by zero (cost is already clamped)
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf (should not happen), return zeros
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
        
    return scores
# EVOLVE-BLOCK-END
