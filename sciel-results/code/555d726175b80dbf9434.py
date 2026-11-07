import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from the posterior
    # We need a deterministic score for the 2000 candidates, 
    # but TS usually involves a single sample per decision step.
    # Here, we are scoring 2000 candidates to pick ONE.
    # Standard TS: Draw one sample f_sample(x) for each x? 
    # No, usually you draw one sample from the GP posterior (a function) 
    # and then minimize it. 
    # Since we don't have the GP kernel, we can't sample the function.
    # We only have pointwise mu and sigma.
    # So we approximate by sampling independently at each point:
    # f_sample_i = mu_i + sigma_i * z_i
    
    # However, independent sampling ignores the correlation between points.
    # This is a common approximation when only pointwise predictions are available.
    
    z = np.random.normal(0, 1, size=len(mu))
    f_sampled = mu + sigma_safe * z
    
    # Cost penalty
    # Parent used power 1.5. Let's stick with a similar structure.
    cost_penalty = np.power(cost, 1.5)
    
    # We want to minimize f_sampled * cost_penalty? 
    # Or minimize f_sampled / cost_penalty?
    # "Evaluating x costs...". Usually cost is additive in the objective if it's a budget constraint.
    # But the parent used division and got a good score.
    # Division favors cheap points more aggressively.
    
    score = -f_sampled / cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
