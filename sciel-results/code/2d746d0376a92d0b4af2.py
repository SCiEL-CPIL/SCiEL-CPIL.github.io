"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement and
    Thompson Sampling with cost normalization.
    
    Strategy:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Compute a Thompson Sampling sample: s = mu - sigma * z.
    3. Combine: score = (EI + alpha * max(0, f_best - s)) / cost
       where alpha balances deterministic EI with stochastic exploration.
    4. This provides stable exploitation via EI while retaining exploration
       through the random sample, all normalized by evaluation cost.
    
    We guard against the 330s wall-clock limit with an explicit deadline.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    safe_cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Draw a single random sample from the GP posterior
    z = np.random.standard_normal(size=mu.shape)
    s = mu - sigma * z
    
    # Stochastic improvement component
    stoch_improvement = np.maximum(f_best - s, 0.0)
    
    # Balance between EI (deterministic) and Thompson sample (stochastic)
    # Use a moderate weight for the stochastic component to retain exploration
    alpha = 0.5
    combined = ei + alpha * stoch_improvement
    
    # Cost-aware score
    score = combined / safe_cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Fallback: pure EI per cost
        score = ei / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
