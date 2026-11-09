"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a UCB-like strategy.
    We minimize the function, so we want low mu and high sigma (exploration).
    We penalize high cost.
    
    Score = -mu + beta * (sigma / cost)
    
    We use a fixed beta that balances exploitation and exploration.
    Since the problem is 8D and we have a limited budget, we want to explore
    efficiently. A higher beta encourages exploration of cheap, uncertain regions.
    """
    # Safety check for time
    if time.time() > _DEADLINE:
        # Return a deterministic fallback if we are running out of time
        # Prefer low cost and low mu
        return -mu / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any non-finite values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-6)
    
    # UCB for minimization: -mu + beta * sigma
    # Cost-aware: divide exploration term by cost
    # We can also normalize mu by cost if we want, but typically we just
    # want the exploration to be cost-efficient.
    
    # Let's use a beta that might adapt slightly or is fixed.
    # A common heuristic for UCB in BO is beta ~ sqrt(log(n)) but here we don't have n easily.
    # Let's use a fixed beta. 2.0 is a reasonable starting point for 8D.
    beta = 2.0
    
    # Exploration term: sigma / cost
    # Exploitation term: -mu
    # We might also want to penalize high cost in the exploitation term?
    # If a point is very good (low mu) but very expensive, is it worth it?
    # The task says "cost-aware". Usually this means we prefer cheaper points
    # if the expected improvement is similar.
    
    # Let's try: Score = (-mu - gamma * cost) + beta * (sigma / cost)
    # Or simpler: Score = -mu / cost + beta * sigma / cost?
    # If we divide mu by cost, we are saying "improvement per unit cost".
    # Let's stick to the previous finding that sigma/cost helped.
    
    # Strategy: 
    # 1. Exploitation: -mu (we want low values)
    # 2. Exploration: sigma / cost (we want high uncertainty per cost)
    # 3. Cost penalty: -alpha * cost (we want low cost)
    
    # Let's combine:
    # score = -mu + beta * (sigma / cost) - alpha * cost
    
    # Tuning alpha and beta.
    # If alpha is too high, we only look at cheap points.
    # If beta is too high, we look at noisy points.
    
    alpha = 0.1
    beta = 2.0
    
    score = -mu + beta * (sigma / cost) - alpha * cost
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
