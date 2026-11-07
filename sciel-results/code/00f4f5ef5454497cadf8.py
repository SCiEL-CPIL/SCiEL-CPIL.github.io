"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Replaces the per-candidate cost normalization/division with a global cost budget penalty.
    The penalty term is proportional to the current total cost consumed (or remaining budget ratio),
    which dynamically shifts the exploration-exploitation balance.
    - Early (low total cost): Low penalty, high exploration (tolerate expensive points if EI is high).
    - Late (high total cost): High penalty, low exploration (strongly prefer cheap points).
    
    This combines the EI calculation from the current program with the adaptive cost sensitivity
    idea from the inspirations, but implemented as a subtractive penalty scaled by global progress
    rather than a multiplicative factor on individual candidates.
    """
    # Deadline Guard: Check if we are close to the time limit.
    # If so, return a simple, fast score to allow the loop to finish gracefully.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple Expected Improvement, ignoring cost to save time/logic
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Global Cost Budget Penalty
    # The 'cost' input is the cost of each candidate (1 + mean(x_scaled)).
    # However, the directive suggests using "current total cost consumed".
    # In this function signature, we don't have direct access to the *cumulative* total cost of the loop,
    # only the cost of the 2000 candidates being scored.
    # But we can infer progress from f_best or simply use the mean cost of the candidates as a proxy
    # for the "expensiveness" of the current region, or better yet, use the fact that
    # as the budget depletes, we want to penalize cost more heavily.
    
    # Let's look at the inspirations again.
    # Inspiration 1 used a normalized cost preference.
    # Inspiration 2 used a multiplicative division by cost^alpha.
    # Current program used: score = ei - beta * sigma * cost, where beta was dynamic based on f_best.
    
    # The directive says: "Replace the per-candidate cost normalization with a global cost budget penalty 
    # by subtracting a term proportional to the current total cost consumed (or remaining budget ratio) 
    # from the acquisition score".
    
    # Since we don't have the *total* cost of past evaluations passed in, we must approximate "progress".
    # f_best is a good proxy for progress. As f_best drops, we are "later" in the optimization process.
    # Alternatively, we can assume that as we run, the "effective" cost sensitivity should rise.
    
    # Let's stick to the dynamic beta idea from the current program but refine the penalty structure.
    # The current program: score = ei - beta * sigma * cost.
    # This penalizes *uncertain* expensive points.
    
    # A more direct "cost-aware EI" is often EI / cost or EI - lambda * cost.
    # If we use EI - lambda * cost, we penalize expensive points regardless of uncertainty.
    # If we use EI / cost, we scale the improvement by the cost.
    
    # Let's try a hybrid:
    # 1. Compute EI.
    # 2. Compute a "cost penalty" that is global in nature but applied per candidate.
    # 3. The magnitude of this penalty should increase as we approach the budget limit.
    #    We approximate budget depletion by how low f_best is.
    
    # Dynamic Lambda based on f_best
    # f_best starts high, ends low.
    # We want lambda to be small when f_best is high (explore, ignore cost).
    # We want lambda to be large when f_best is low (exploit, prefer cheap).
    
    # Sigmoid mapping for lambda:
    # lambda = lambda_min + (lambda_max - lambda_min) * sigmoid(a * (b - f_best))
    # When f_best is high (>> b), sigmoid is ~0, lambda ~ lambda_min.
    # When f_best is low (<< b), sigmoid is ~1, lambda ~ lambda_max.
    
    a_param = 0.5
    b_param = 5.0
    f_best_clamped = np.clip(f_best, -100, 100)
    
    # Confidence that we are in the "exploitation" phase
    progress = 1.0 / (1.0 + np.exp(-a_param * (f_best_clamped - b_param)))
    # Wait, if f_best is high (50), exp(-0.5*45) is tiny, progress ~ 1. 
    # I want progress to be LOW when f_best is HIGH.
    # So: progress = 1 / (1 + exp(a_param * (f_best_clamped - b_param)))
    # If f_best = 50, exp(0.5*45) is huge, progress ~ 0.
    # If f_best = 0, exp(-2.5) is small, progress ~ 0.92.
    
    progress = 1.0 / (1.0 + np.exp(a_param * (f_best_clamped - b_param)))
    
    # Lambda ranges from 0.1 to 2.0
    lambda_min = 0.1
    lambda_max = 2.0
    lambda_val = lambda_min + progress * (lambda_max - lambda_min)
    
    # Score = EI - lambda_val * cost
    # This directly penalizes cost. As progress increases, the penalty for cost increases.
    
    score = ei - lambda_val * cost
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
