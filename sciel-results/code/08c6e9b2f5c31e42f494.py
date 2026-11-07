"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 2.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Strategy: Use a hybrid of Expected Improvement (EI) and a "Cost-Adjusted Exploration" term.
    # The current seed uses EI / cost^2. 
    # The problem is that EI can be zero for points where mu >= f_best, even if sigma is high (high uncertainty).
    # This causes the algorithm to ignore potentially promising unexplored regions if the mean prediction is not yet better than the best found.
    # 
    # A different strategy: "Confidence-Weighted Improvement"
    # Score = (f_best - mu + kappa * sigma) / cost
    # This is similar to UCB but scaled by cost. 
    # However, UCB can be too aggressive in exploration.
    # 
    # Let's try a "Probability of Improvement" (PI) based approach combined with cost.
    # PI = P(f(x) < f_best) = norm.cdf((f_best - mu) / sigma)
    # This is 0 when mu >> f_best and 1 when mu << f_best.
    # It doesn't account for the magnitude of improvement, only the probability.
    # 
    # Let's stick to EI but modify the exploration/exploitation balance.
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # 
    # Alternative: Use "Expected Hypervolume Improvement" (EHI) approximation? Too complex.
    # 
    # Let's try a different cost model. 
    # The cost is 1 + mean(x). 
    # Instead of dividing by cost, we can subtract a penalty.
    # Score = EI - lambda * cost
    # But EI can be negative? No, EI is always >= 0.
    # If EI is small, the penalty might dominate.
    # 
    # Let's look at the "Different Strategy" requirement again.
    # Current: EI / cost^2.
    # 
    # New Idea: "Adaptive Kriging Distance"
    # Since we don't have coordinates, we can't compute true KD.
    # But we can use sigma as a proxy for "distance to information".
    # 
    # Let's try: Score = (EI + alpha * sigma * (1 - cost/max_cost)) / cost
    # This adds an explicit exploration bonus that is higher for cheaper points.
    # 
    # Actually, let's simplify. The main weakness of EI/cost is that it might get stuck in local optima if the GP is confident (low sigma) but wrong (mu > f_best).
    # If mu > f_best, EI is small. 
    # If we add a term that rewards high sigma regardless of mu, we force exploration.
    # 
    # Score = EI / cost + beta * sigma / cost
    # This is equivalent to (EI + beta * sigma) / cost.
    # EI already contains a sigma term: sigma * phi(z).
    # phi(z) is max at z=0 (mu = f_best). 
    # If mu < f_best (improvement expected), z > 0, phi(z) decreases.
    # If mu > f_best (no improvement expected), z < 0, phi(z) decreases.
    # 
    # So EI naturally balances exploration.
    # 
    # What if the issue is the cost exponent? 
    # Current: cost^2.
    # Maybe cost^1 is better? Or cost^0.5?
    # 
    # Let's try a "Cost-Discounted EI" with a different exponent.
    # Score = EI / (cost ** 0.5)
    # This penalizes high cost less aggressively, allowing the algorithm to explore more expensive regions if the EI is high enough.
    # This might help escape local optima in expensive regions.
    
    # Let's implement Score = EI / (cost ** 0.5)
    # This is a materially different cost model (sub-linear penalty vs quadratic).
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    score = ei / np.sqrt(cost_safe)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
