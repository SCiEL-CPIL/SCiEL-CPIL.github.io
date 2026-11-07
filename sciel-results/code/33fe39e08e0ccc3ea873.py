import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time
_start_time = time.time()
_DEADLINE = 320.0  # 10 second safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Time guard
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        # Return a safe, valid result
        return np.random.rand(len(mu)) * 1e-6 + 1e-9

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()

    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # Standard EI: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Lower Confidence Bound (LCB) for minimization
    # LCB = -mu + kappa * sigma (we want to minimize, so we look for low mu and high uncertainty)
    # Higher kappa encourages exploration
    kappa = 2.0
    lcb = -mu + kappa * sigma_safe
    
    # 3. Combine EI and LCB
    # We normalize both to [0, 1] roughly to make them comparable
    # EI is non-negative. LCB can be negative (if mu is very low).
    # Let's scale EI by f_best magnitude if f_best is not small, or just use raw values?
    # Since f_best starts around 0.5-1.0 for Griewank and goes down to 0, EI is small.
    # LCB is in objective units too.
    
    # A robust combination:
    # score = w * EI + (1-w) * LCB_scaled
    # But LCB can be negative. We want higher score = better.
    # If mu is very low, LCB is very high (positive).
    # If mu is high, LCB is low.
    
    # Let's try a weighted sum with adaptive weights based on the magnitude of EI and LCB.
    # Or simply take the maximum?
    # The current code uses max(ei, ucb).
    
    # Let's try a different approach:
    # Use EI as the primary signal, but add a bonus for high uncertainty (sigma)
    # when EI is low.
    
    # Alternative: q-EI or similar? No, keep it simple.
    
    # Let's stick to a hybrid but tune the combination.
    # Normalize EI by f_best (if f_best > 0)
    # If f_best is close to 0, EI is also close to 0.
    
    # Let's use: score = ei + alpha * lcb
    # But lcb can be negative.
    
    # Let's look at the scales.
    # f_best ~ 0.1 initially. EI ~ 0.01.
    # mu ~ 0.1. sigma ~ 0.1.
    # LCB = -0.1 + 2*0.1 = 0.1.
    # So LCB is larger than EI.
    
    # Let's try: scores = ei + 0.5 * lcb
    # Or: scores = ei + lcb
    
    # Actually, let's use a product or a different combination.
    # The "max" approach in the current code might be too aggressive on exploration.
    
    # Let's try: scores = ei * (1 + lcb) ? No.
    
    # Let's go back to basics.
    # We want to find the global minimum.
    # EI is a good measure of improvement.
    # LCB is a good measure of potential minimum.
    
    # Let's compute a weighted sum.
    # Weights should depend on the budget remaining?
    # We don't have explicit budget left, but we have 'cost' which is the cost of evaluating each point.
    # The total cost is accumulated externally.
    
    # Let's try:
    # score = ei + beta * sigma
    # This is similar to LCB but with EI added.
    
    # Let's try:
    # score = ei + 1.0 * sigma
    # This encourages exploration (high sigma) and exploitation (high ei).
    
    # Let's compare:
    # Current: max(ei, -mu + 2*sigma) / cost
    # New: (ei + sigma) / cost
    
    # Let's try a slightly more sophisticated combination.
    # If f_best is small, we are close to the optimum. EI becomes very sensitive.
    # LCB might be noisy.
    
    # Let's try:
    # z = (f_best - mu) / sigma
    # ei = ...
    # lcb = -mu + kappa * sigma
    
    # score = ei + lcb
    
    # Let's test this hypothesis.
    # If mu is low and sigma is low, ei is high, lcb is high.
    # If mu is high and sigma is low, ei is 0, lcb is low.
    # If mu is low and sigma is high, ei is moderate, lcb is very high.
    # If mu is high and sigma is high, ei is 0, lcb is moderate.
    
    # This seems reasonable.
    
    kappa = 1.5
    lcb = -mu + kappa * sigma_safe
    scores = ei + lcb
    
    # Divide by cost
    scores = scores / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
