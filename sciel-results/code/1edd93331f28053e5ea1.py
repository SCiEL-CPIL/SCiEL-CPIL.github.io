import numpy as np
import time
from scipy.stats import norm

# Global start time to track overall progress
_START_TIME = time.time()
# Safety margin before the 330s hard limit
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using adaptive exploration-exploitation balance.
    
    Structural change: Instead of a fixed EI/cost formula, uses a time-adaptive
    mixture of UCB (exploration) and EI (exploitation) where the balance shifts
    from exploration-heavy early in the budget to exploitation-heavy late.
    """
    # Deadline guard: If we are running out of time, switch to pure exploitation
    if time.time() > _DEADLINE:
        scores = -mu
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize sigma and cost
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    # Estimate progress through budget based on f_best and cost
    # Total budget is 30, initial 5 random points cost ~5 * 1.5 = 7.5 avg
    # So we've used roughly cost_sum, remaining = 30 - cost_sum
    # But we don't have cost_sum directly; use f_best as a proxy for progress
    # Better: use the mean cost as a proxy for how many evals we've done
    # Actually, we can estimate: if f_best is very low, we're late.
    # Simpler: use a fixed schedule based on the typical number of iterations.
    
    # Budget: 30 total. Initial 5 points cost ~5*1.5=7.5 on average.
    # Remaining budget ~22.5. Each eval costs ~1.5 avg, so ~15 more evals.
    # Let's estimate iteration count from f_best magnitude or use a simple heuristic.
    
    # Adaptive beta: start high (exploration), decay to low (exploitation)
    # Use f_best as a proxy: when f_best is still high, we're early
    # When f_best is close to 0, we're late
    # Rosenbrock min is 0, so f_best in [0, large]
    
    # Normalize f_best: early if f_best > 10, late if f_best < 1
    # Use a smooth transition
    f_best_norm = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # Beta for UCB: higher when exploring (early), lower when exploiting (late)
    # Early: beta ~ 2.0, Late: beta ~ 0.5
    beta = 0.5 + 1.5 * f_best_norm
    
    # Calculate UCB
    ucb = -mu + beta * sigma
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI to be comparable to UCB scale
    # EI is in objective units, UCB is in objective units
    # But they have different scales. Let's use a weighted sum.
    
    # Weight: early -> more UCB, late -> more EI
    # w_ucb = f_best_norm, w_ei = 1 - f_best_norm
    w_ucb = 0.3 + 0.7 * f_best_norm  # early: 1.0, late: 0.3
    w_ei = 1.0 - w_ucb               # early: 0.0, late: 0.7
    
    # Combine: use a geometric-like combination to avoid scale issues
    # Scale EI by typical improvement magnitude
    typical_improvement = max(f_best, 1e-6)
    ei_scaled = ei / typical_improvement  # normalize to [0, ~1]
    
    # UCB normalized: center around 0
    ucb_scaled = ucb / (1.0 + np.abs(mu).max() if mu.size > 0 else 1.0)
    
    # Simpler: just use weighted sum with proper scaling
    # EI is already in objective units, UCB is in objective units
    # The issue is their ranges differ. Let's use:
    # score = w_ucb * ucb + w_ei * ei
    # But divide by cost to account for cost-awareness
    
    scores = (w_ucb * ucb + w_ei * ei) / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
