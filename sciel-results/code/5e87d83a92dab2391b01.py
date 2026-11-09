"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_DEADLINE = None
def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290  # 290s safety margin from 300s limit
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Uses a structural change: instead of mixing Thompson Sampling with EI,
    we use a cost-normalized Expected Improvement with adaptive exploration
    via a variance-aware penalty that encourages visiting high-uncertainty
    regions when the current best is far from the known minimum (0).
    
    The key structural change: we model the "remaining improvement potential"
    as the gap to the global minimum (0), not just to f_best. This makes
    exploration more aggressive early and exploitation sharper late.
    
    Parameters:
    - mu: (2000,) posterior mean in objective units
    - sigma: (2000,) posterior std in objective units  
    - f_best: scalar, lowest observed value
    - cost: (2000,) evaluation cost for each candidate
    
    Returns:
    - (2000,) scores, higher = evaluate next
    """
    t_start = time.time()
    deadline = _get_deadline()
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute Expected Improvement relative to f_best
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Structural change: also compute "expected gap to zero" which drives
    # exploration when f_best is still far from the true minimum (0)
    # E[min(0, f(x))] = -E[max(0, -f(x))]
    # For a normal N(mu, sigma^2): E[max(0, -f)] = -mu*Phi(-mu/sigma) + sigma*phi(-mu/sigma)
    # So E[min(0, f)] = mu*Phi(-mu/sigma) - sigma*phi(-mu/sigma)
    # But we want the expected improvement toward 0, which is E[(-f)^+] = E[max(0, -f)]
    # This is the expected value we'd gain by going to 0 if we could.
    
    # A simpler structural approach: use a "regret to zero" formulation
    # The expected improvement toward the global min (0) is:
    # EI_zero = E[max(0, f(x) - 0)] is not useful for minimization.
    # Instead, consider: how much can we expect to reduce f_best?
    # That's just EI as computed above.
    
    # The real structural change: use a cost-aware score that combines
    # (1) immediate improvement (EI/cost) and 
    # (2) information gain proxy (sigma/cost) weighted by remaining budget
    
    # Estimate remaining budget: we don't know it directly, but we can use
    # the fact that cost is ~1 to 2, and we have ~30 total budget.
    # Use sigma as a proxy for information value.
    
    # Cost-normalized EI
    ei_per_cost = ei / (cost + 1e-9)
    
    # Information value: high sigma means we learn a lot. But we weight it
    # by how far we are from the minimum. When f_best is close to 0, 
    # information is less valuable.
    # Weight = f_best / (f_best + 1) is 0 when f_best=0, 1 when f_best is large
    info_weight = f_best / (f_best + 1.0)
    info_per_cost = sigma / (cost + 1e-9) * info_weight
    
    # Combine with a fixed but principled weight
    # Early on (f_best large), explore more. Late (f_best small), exploit.
    alpha = 0.3  # weight on information gain
    
    score = ei_per_cost + alpha * info_per_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline - 5:  # 5s safety margin
        # Return a simple fallback: prefer low mu and low cost
        fallback = (f_best - mu) / (cost + 1e-9)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    return score
# EVOLVE-BLOCK-END
