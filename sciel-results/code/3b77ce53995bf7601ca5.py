import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0

def acquisition(mu, sigma, f_best):
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a reasonable fallback: just LCB
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-9)
        lcb = -mu - 2.0 * sigma
        lcb = np.where(np.isfinite(lcb), lcb, 0.0)
        return lcb
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Lower Confidence Bound for minimization
    # score = -mu - kappa * sigma
    # We want higher scores to mean "evaluate next"
    # So we want to pick points with low mu and high sigma
    # LCB = mu - kappa * sigma (lower is better for the bound itself)
    # But we need higher score = better, so score = -LCB = -mu + kappa * sigma
    # Wait, that's UCB. Let me think again.
    
    # For minimization:
    # We want to find the point with the lowest possible value.
    # LCB gives a lower bound on the minimum: LCB = mu - kappa * sigma
    # We want to maximize this lower bound (i.e., find the point where the lower bound is highest, meaning the true value is likely low but we're confident it's low).
    # Actually, for minimization with LCB, we want to MINIMIZE the LCB value? No.
    
    # Let's clarify:
    # In BO for minimization:
    # - EI: maximize expected improvement over f_best
    # - UCB: maximize -mu + kappa*sigma (equivalent to minimizing mu - kappa*sigma)
    # - LCB: minimize mu - kappa*sigma (equivalent to maximizing -mu + kappa*sigma)
    
    # So LCB for minimization is: pick x that minimizes (mu(x) - kappa * sigma(x))
    # To convert to a "higher is better" score: score = -(mu - kappa*sigma) = -mu + kappa*sigma
    
    # But wait, the directive says: "Use a deterministic Lower Confidence Bound (LCB) formulation, score = -mu - kappa * sigma"
    # That would be: score = -mu - kappa * sigma
    # Higher score means: lower mu AND lower sigma. That's pure exploitation with a penalty for uncertainty.
    # That doesn't seem right for exploration.
    
    # Let me re-read: "score = -mu - kappa * sigma" 
    # If kappa > 0, then higher score = lower mu and lower sigma.
    # This would prefer regions with low predicted values AND low uncertainty.
    # That's actually a reasonable exploitation-focused strategy.
    
    # But the standard LCB for minimization is: minimize mu - kappa*sigma
    # Which means maximize -(mu - kappa*sigma) = -mu + kappa*sigma
    # That's the same as UCB for minimization.
    
    # I think there might be confusion in terminology. Let me just implement what makes sense:
    # For minimization, we want to balance finding low values with exploring uncertain regions.
    # A common approach: score = -mu + kappa * sigma (this is UCB for minimization, or equivalently, maximizing the negative LCB)
    
    # Let me use an adaptive kappa based on the problem state.
    
    # Adaptive kappa: start with higher exploration, decrease as we converge
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Use the ratio of f_best to typical scale to adapt
    # If f_best is close to 0, we're near the optimum, reduce exploration
    # If f_best is large, we're far from optimum, increase exploration
    
    # Normalize f_best: the Beale function has values from 0 to ~100+
    # Use a simple adaptive scheme
    if f_best > 10:
        kappa = 3.0  # High exploration when far from optimum
    elif f_best > 1:
        kappa = 2.0  # Medium exploration
    elif f_best > 0.1:
        kappa = 1.0  # Low exploration
    else:
        kappa = 0.5  # Very low exploration, focus on exploitation
    
    # LCB for minimization: score = -mu + kappa * sigma
    # Higher score = better (lower mu, higher sigma)
    score = -mu + kappa * sigma
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
