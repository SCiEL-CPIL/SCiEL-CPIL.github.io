import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Base Score: EI / cost^2 (Strong exploitation + strong cost penalty)
    # This matches the structure of the highest-scoring inspiration
    base_score = ei / (cost ** 2)
    
    # Pareto Front Calculation
    # A point i dominates point j if:
    # 1. ei[i] >= ei[j]
    # 2. cost[i] <= cost[j]
    # 3. ei[i] > ei[j] OR cost[i] < cost[j] (strict in at least one)
    
    # Vectorized comparison
    # ei shape: (N,), cost shape: (N,)
    # We need to compare all pairs (i, j)
    # ei_i = ei[:, None] (N, 1)
    # ei_j = ei[None, :] (1, N)
    
    ei_i = ei[:, None]
    ei_j = ei[None, :]
    cost_i = cost[:, None]
    cost_j = cost[None, :]
    
    # Condition 1 & 2: Non-inferiority
    non_inferior = (ei_i >= ei_j) & (cost_i <= cost_j)
    
    # Condition 3: Strict improvement in at least one dimension
    strict_ei = ei_i > ei_j
    strict_cost = cost_i < cost_j
    strict = strict_ei | strict_cost
    
    # Dominance matrix: D[i, j] is True if i dominates j
    dominates = non_inferior & strict
    
    # Count how many points dominate each point j
    # sum over axis 0 (rows i) to get count for each column j
    dom_count = np.sum(dominates, axis=0)
    
    # Pareto Factor: 1.0 for non-dominated (count 0), decays for dominated
    # Using 1 / (1 + count) provides a smooth decay
    # Points on the front get factor 1.0
    # Points dominated by 1 point get 0.5
    # Points dominated by 2 points get 0.33
    pareto_factor = 1.0 / (1.0 + dom_count)
    
    # Final Score
    score = base_score * pareto_factor
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
