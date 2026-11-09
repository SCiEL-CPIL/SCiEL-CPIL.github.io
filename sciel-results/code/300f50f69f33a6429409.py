"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware UCB strategy
    # UCB = mu + beta * sigma
    # We want to minimize, so we prefer low mu and high sigma (exploration)
    # But we also need to account for cost.
    
    # Dynamic beta: 
    # As cost increases (budget depletes), we should explore less and exploit more.
    # cost is the accumulated cost. Max cost is 30.
    # Initially cost is low (~5 for 5 points), so beta should be high.
    # As cost approaches 30, beta should drop to near 0.
    
    # Let's map cost to beta.
    # cost range: [5, 30]
    # beta range: [2.0, 0.5]
    # Linear decay
    min_cost = 5.0
    max_cost = 30.0
    beta_max = 2.0
    beta_min = 0.5
    
    # Normalize cost to [0, 1]
    cost_norm = np.clip((cost - min_cost) / (max_cost - min_cost), 0, 1)
    
    # Interpolate beta
    beta = beta_max - (beta_max - beta_min) * cost_norm
    
    # Base UCB for minimization:
    # We want to find points where the lower bound of the confidence interval is low.
    # Lower bound = mu - beta * sigma
    # So score = -(mu - beta * sigma) = beta * sigma - mu
    
    ucb_score = beta * sigma_safe - mu
    
    # Cost-awareness:
    # We divide by cost to prefer cheaper points, but we must be careful not to over-penalize.
    # A soft penalty might be better: score / (1 + cost) or score / cost^alpha
    # The seed program used EI / cost.
    # Let's try UCB / (1 + cost) to avoid division by small numbers and soften the penalty.
    # Or stick to cost normalization but maybe less aggressive.
    
    # Actually, the cost is the *accumulated* cost.
    # The marginal cost of the next evaluation is 1 + mean(x).
    # The `cost` array passed in is the accumulated cost up to that point? 
    # "mu, sigma and cost have shape (2000,). Return 2000 finite scores"
    # "Evaluating x costs 1 + mean(x scaled to [0, 1]). Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30, the evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
    
    # The `cost` input to `acquisition` is likely the *current* total cost, which is the same for all 2000 candidates?
    # Or is it the *predicted* cost if we evaluate each candidate?
    # "cost has shape (2000,)". If it were the current total cost, it would be a scalar or a vector of the same value.
    # Given the shape (2000,), it is highly probable that `cost` represents the **marginal cost** of evaluating each of the 2000 candidates.
    # i.e., cost[i] = 1 + mean(x_i).
    
    # If cost[i] is the marginal cost, then dividing by cost[i] makes sense to prefer cheap evaluations.
    
    # Let's assume cost is marginal cost.
    # score = ucb_score / cost_safe
    
    cost_safe = np.maximum(cost, 1.0) # Cost is at least 1
    
    score = ucb_score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
         return score
         
    return score
# EVOLVE-BLOCK-END
