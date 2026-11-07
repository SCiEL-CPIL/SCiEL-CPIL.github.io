"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of calculating Expected Improvement (EI) and dividing by cost,
    we sample from the posterior distribution N(mu, sigma^2) for each candidate.
    The candidate with the lowest sampled value is selected. This implicitly
    balances exploration and exploitation. To incorporate cost, we can penalize
    the sampled value slightly or rely on the fact that cheaper points are
    evaluated more often in the loop structure if we were to implement a
    full BO-NC, but here we stick to a simple, robust TS approach.
    
    However, the task requires returning scores for 2000 candidates where higher
    is better. In Thompson Sampling, we typically pick argmin(sampled_values).
    To convert this to a score where higher is better, we can simply return
    -sampled_values. But we must be careful: if we just return -sample, the
    "best" score corresponds to the lowest sample.
    
    The cost is given. A pure TS doesn't use cost in the scoring function itself
    unless we modify the sample. A common cost-aware TS variant adds a cost penalty
    to the sample: sample + lambda * cost. Then we minimize this.
    So score = -(sample + lambda * cost).
    
    Let's use a small lambda to account for cost, or just use pure TS if the
    cost variation is not extreme. The cost is 1 + mean(x_scaled).
    Range of cost: [1, 2].
    Range of f: [0, ~20] for Powell 4D.
    
    If we use pure TS (score = -sample), the cost is ignored in the selection
    criterion, but the loop structure "evaluates the best one" will just pick
    the lowest sample. Since the cost varies between 1 and 2, ignoring it might
    be suboptimal if cheap points are significantly better.
    
    Let's try a weighted combination: score = -sample / cost.
    This is similar to EI/cost but uses the sample instead of EI.
    Actually, EI is E[max(f_best - f, 0)]. TS samples f.
    Minimizing f is the goal.
    
    Let's use: sample = mu + sigma * z, where z ~ N(0,1).
    We want to minimize sample.
    To make it cost-aware, we can minimize sample + penalty * cost.
    Or simply divide by cost? Dividing a noisy sample by cost is not standard.
    
    Standard Cost-Aware TS:
    Sample y ~ N(mu, sigma^2).
    Select x that minimizes y + lambda * c(x).
    
    Let's estimate lambda. The range of y is large. The range of cost is small [1,2].
    If lambda is too large, we always pick the cheapest point (x=0).
    If lambda is 0, we ignore cost.
    
    Let's try a simple heuristic: score = -sample.
    Wait, the prompt says "cost-aware". The current program uses EI/cost.
    If I switch to TS, I should probably incorporate cost.
    
    Let's try: score = -sample / cost.
    This favors points with low samples AND low cost.
    
    Args:
        mu: Mean predictions, shape (N,)
        sigma: Standard deviation predictions, shape (N,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point, shape (N,)
        
    Returns:
        Scores for each candidate, shape (N,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is safe for division
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random normal samples
    # Using a fixed seed per call? No, the evaluator calls this in a loop.
    # We need randomness. np.random.normal is fine.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from posterior
    samples = mu + sigma * z
    
    # Cost-aware scoring:
    # We want to minimize samples.
    # To account for cost, we can penalize expensive points.
    # score = -samples / cost_safe seems like a reasonable proxy for "value per cost".
    # However, dividing a random variable by cost changes the distribution shape.
    # Alternative: samples + lambda * cost.
    # Let's try samples / cost_safe first, as it mirrors the EI/cost structure but with samples.
    
    scores = -samples / cost_safe
    
    # Replace non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
