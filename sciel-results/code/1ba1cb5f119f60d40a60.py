"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIMEOUT_LIMIT = 320.0  # Safety margin before 330s kill

def _check_deadline():
    return (time.time() - _start_time) > _TIMEOUT_LIMIT

def _safe_div(a, b, default=0.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(b != 0, a / b, default)
    return np.nan_to_num(res, nan=default, posinf=default, neginf=default)

def acquisition(mu, sigma, f_best, cost):
    # Check for deadline to avoid timeout
    if _check_deadline():
        # Return a valid fallback: pure exploitation (lowest mu)
        # Higher score is better, so we want high score for low mu.
        # Return -mu.
        return -mu

    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = _safe_div(f_best - mu, sigma_safe, 0.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB)
    # UCB = mu + kappa * sigma
    # We want to minimize, so we prefer low mu and high sigma (exploration).
    # The score should be higher for better candidates.
    # Standard UCB for minimization: score = -mu + kappa * sigma? 
    # Actually, let's stick to the standard form for maximization of "regret reduction" or similar.
    # For minimization, we want to find low values.
    # Let's define UCB score as -mu + kappa * sigma.
    # If mu is low (good), -mu is high.
    # If sigma is high (uncertain), the bonus is high.
    
    # Dynamic Kappa
    # Total budget is 30.
    # Current cost is the sum of costs of evaluated points? 
    # The input 'cost' has shape (2000,). This is likely the cost of evaluating each candidate x.
    # Wait, the prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # "Runs start with 5 uniform random points (cost counted)."
    # "While total cost is below 30..."
    # The function receives `cost` with shape (2000,). This is the cost of the *candidates*.
    # But we don't know the *current total cost* from the arguments directly.
    # However, we can infer progress. 
    # Actually, the prompt doesn't pass the current iteration count or total cost used so far.
    # It only passes `cost` for the 2000 candidates.
    # How do we know how much budget is left?
    # We can't directly. But we can assume that as the optimization progresses, f_best decreases.
    # Or we can use a fixed schedule based on... nothing?
    
    # Let's look at the inspirations. 
    # Inspiration 2: "Dynamic Budget Decay"
    # Inspiration 3: "phase-dependent exploration coefficient (kappa)"
    
    # Since we don't have the iteration count, we might have to rely on f_best or just a fixed, well-tuned kappa.
    # Or, we can assume that the "cost" array contains the cost of the candidates, and maybe we can estimate
    # the "phase" by looking at the spread of mu/sigma? No, that's unreliable.
    
    # Let's try a fixed but optimized kappa for UCB, and blend with EI.
    # EI is generally better for convergence. UCB is better for exploration.
    # A common hybrid is: score = alpha * EI + (1-alpha) * UCB
    # Or simply: score = EI / cost + beta * (sigma / cost)
    
    # Let's refine the cost-awareness.
    # Cost-aware EI: EI / cost
    # Cost-aware UCB: (-mu + kappa * sigma) / cost
    
    # Let's try a dynamic kappa based on the assumption that early on, f_best is high (far from 0), 
    # and later f_best is low.
    # If f_best is close to 0, we should exploit more.
    # If f_best is far from 0, we should explore more.
    
    # The minimum of the function is 0.
    # So f_best <= 0? No, "min 0" means the global minimum is 0.
    # So f_best >= 0.
    # As f_best -> 0, we are close to the optimum.
    
    # Let's define a "closeness" factor.
    # If f_best is small, reduce exploration.
    # kappa = base_kappa * exp(-f_best / scale)
    
    # Let's pick base_kappa = 1.0 and scale = 1.0 (arbitrary, needs tuning but let's try).
    # Actually, the function values for 3-hump camel can be quite large initially.
    # Let's just use a fixed kappa for now, as we don't have a robust phase indicator.
    # But wait, the prompt says "cost-aware".
    
    # Let's try:
    # score = (EI + kappa * sigma) / cost
    # This combines exploitation (EI) and exploration (sigma) and penalizes expensive points.
    
    # Let's set kappa = 0.5 initially.
    
    kappa = 0.5
    
    ucb_term = -mu + kappa * sigma_safe
    ei_term = ei
    
    # Blend them
    # EI is in objective units. UCB term is in objective units.
    # We want to maximize the score.
    # score = (ei_term + ucb_term) / cost
    
    # Ensure cost is positive and safe
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (ei_term + ucb_term) / cost_safe
    
    # Ensure finiteness
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
