"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290.0  # 10s safety margin
    return _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    deadline = _get_deadline()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical stability: avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Check deadline
    if time.time() > deadline - 1.0:
        # Return a safe fallback: prefer low cost, high uncertainty
        # Simple heuristic: 1/cost * sigma
        scores = sigma_safe / cost_safe
        return scores
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = P(Z < z) where Z ~ N(0,1)
    # Note: If mu is much lower than f_best, z is large positive, PI ~ 1.
    # If mu is much higher, z is large negative, PI ~ 0.
    pi = norm.cdf(z)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Structural Change: Use a blend of PI and EI, weighted by cost.
    # PI is more robust to outliers in f_best. EI is more sensitive to the magnitude of improvement.
    # We use Cost-Aware PI as the primary driver, but add a small EI component to favor large improvements.
    # To handle the "local baseline" hint without data, we rely on the GP posterior's uncertainty (sigma)
    # to drive exploration. PI already does this via the z-score.
    
    # Let's try Cost-Aware PI with an uncertainty bonus.
    # Score = PI / cost + beta * (sigma / max_sigma) / cost
    # This explicitly rewards high uncertainty (exploration) per unit cost.
    
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-9:
        max_sigma = 1.0
        
    # Normalize sigma to [0, 1]
    sigma_norm = sigma_safe / max_sigma
    
    # Blend: 80% PI, 20% Uncertainty Bonus
    # The uncertainty bonus encourages exploring unexplored regions (high sigma)
    # even if the current mean is not promising, but penalized by cost.
    exploration_bonus = sigma_norm
    exploitation_score = pi
    
    # Combine
    raw_score = 0.8 * exploitation_score + 0.2 * exploration_bonus
    
    # Cost Awareness
    score = raw_score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() > deadline - 0.5:
        # If we are close to the deadline, just return the calculated score
        # It's already computed efficiently.
        pass
        
    return score
# EVOLVE-BLOCK-END
