"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0 # 5 seconds margin

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are running out of time, return a safe fallback
    # Since this function is fast, this is unlikely to trigger, but good practice.
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
        
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize cost
    cost_safe = np.maximum(cost, 1e-6)
    
    # Exploitation term: Improvement penalized by full cost
    # This discourages evaluating expensive points that are predicted to be good,
    # preferring cheaper points with similar improvement potential.
    improvement = np.maximum(f_best - mu, 0.0)
    exploitation_score = improvement / cost_safe
    
    # Exploration term: Uncertainty penalized by square root of cost
    # This encourages exploring uncertain regions, but less so if they are expensive.
    # The sqrt penalty is milder than the linear penalty, allowing some exploration
    # in high-cost areas if uncertainty is high.
    exploration_score = sigma / np.sqrt(cost_safe)
    
    # Combine scores
    # The sum balances exploitation and exploration.
    scores = exploitation_score + exploration_score
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
