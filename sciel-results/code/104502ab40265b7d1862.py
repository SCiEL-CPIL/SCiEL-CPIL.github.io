"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of computing a deterministic score like EI, we simulate a single
    draw from the posterior distribution of the objective function for each
    candidate. The candidate with the lowest simulated value (adjusted for cost)
    is selected. This naturally balances exploration (high sigma leads to
    potentially lower draws) and exploitation (low mu leads to lower draws).
    
    Args:
        mu: Mean predictions, shape (N,)
        sigma: Standard deviation predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        Scores for each candidate, shape (N,). Higher score means "evaluate this".
        Since Thompson Sampling selects the min of the sample, we return 
        -sampled_value / cost to make it a maximization problem for the caller.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check for deadline. 
    # The problem states "Each evaluation must finish within 300 s".
    # The acquisition function is called inside a loop that runs until cost < 30.
    # However, the acquisition function itself is just scoring 2000 points.
    # The "300s" limit likely applies to the whole run or each call.
    # Given the constraint "evaluator KILLS the program after 330s wall-clock",
    # and that we don't know how much time has elapsed in the parent loop,
    # we should be efficient. 
    # Thompson Sampling is O(N) which is very fast.
    
    # Generate random samples from N(mu, sigma^2)
    # We use a single random number per point to simulate the latent function value.
    # To ensure reproducibility isn't strictly required but stability is,
    # we can use a standard normal random vector.
    
    # Note: In a true GP context, these samples would be correlated.
    # However, the interface only provides mu and sigma (marginal statistics).
    # We assume the marginal distribution is sufficient for this heuristic,
    # which is a standard approximation when full covariance is not passed.
    # This is effectively "independent Thompson Sampling" or "UCB-like" with randomness.
    
    # Draw epsilon ~ N(0, 1)
    eps = np.random.randn(mu.shape[0])
    
    # Simulated function value: y_sim = mu + sigma * eps
    y_sim = mu + sigma * eps
    
    # We want to minimize y_sim, but also account for cost.
    # The current program used EI / cost^exponent.
    # For Thompson Sampling, a common approach is to just pick argmin(y_sim).
    # But we have a cost-aware requirement.
    # Let's define the score as -y_sim / cost.
    # This favors low function values and low costs.
    # However, if y_sim is very low (good), -y_sim is high.
    # If cost is high, the score is lower.
    
    # A more robust cost-aware Thompson sampling might use:
    # score = -y_sim / cost
    # Or perhaps a penalty: score = -y_sim - lambda * cost
    # Given the previous program used division, let's stick to a multiplicative/inverse cost factor.
    
    # To make it "higher is better" for the evaluator:
    # We want to select the point that minimizes y_sim * cost_factor.
    # So the "score" returned should be high for the best candidate.
    # score = - (y_sim / cost)
    
    scores = -y_sim / cost_safe
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
