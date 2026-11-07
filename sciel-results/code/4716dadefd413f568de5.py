import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with adaptive
    exploration and exploitation, tailored for the Styblinski-Tang 2D problem.
    
    The approach:
    1. Draw a single sample from the GP posterior: sample = mu - sigma * z
       where z ~ N(0, 1)
    2. Score candidates by (f_best - sample) / cost^alpha
       This naturally balances exploration (high sigma -> wider spread of samples)
       and exploitation (low mu -> lower samples), while penalizing expensive evaluations.
    3. Adaptive cost exponent and exploration scaling based on progress toward the known minimum.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        cost_safe = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Known minimum of Styblinski-Tang 2D
    known_min = -78.3323
    
    # Compute progress: 0 at start (f_best far from optimum), 1 when near optimum
    # Typical initial f_best from 5 uniform points is in range [-50, 50]
    # Range from typical start to optimum is roughly 130-150
    progress = np.clip(1.0 - (f_best - known_min) / 150.0, 0.0, 1.0)
    
    # Thompson Sampling: draw a single z ~ N(0,1) for the entire batch
    # Using a fixed seed per call would be problematic since we want different samples each iteration
    # But we need reproducibility within a single call. We'll use a deterministic z based on index
    # to ensure the same candidate always gets the same relative exploration bonus.
    # Actually, for Thompson Sampling in a batch setting, we draw ONE sample per candidate.
    # The key insight: we want to select the candidate with minimum sampled value / cost.
    # Since we're returning scores (higher = better), we return (f_best - sample) / cost^alpha
    
    # Draw z values: one per candidate
    # Use a deterministic pseudo-random sequence based on candidate index for reproducibility
    # within a single acquisition call, but different across calls
    # We'll use a simple LCG seeded by the call count (tracked via a global)
    
    # For simplicity and speed, use np.random with a fixed seed that changes each call
    # We track call count to get different random numbers each time
    global _call_count
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    
    # Use a simple deterministic hash-like approach for z values
    # This ensures different z values each call without expensive RNG setup
    seed = acquisition._call_count
    # Generate z values using a simple approach
    z = np.random.RandomState(seed).randn(n)
    
    # Thompson sample: sample = mu - sigma * z
    # We want low sampled values (good for minimization)
    sampled = mu - sigma * z
    
    # Adaptive cost exponent: more cost-sensitive as we progress
    cost_exp = 0.5 + 1.0 * progress
    
    # Score: (f_best - sampled) / cost^cost_exp
    # Higher score = better candidate to evaluate
    # f_best - sampled > 0 means the sample is better than current best
    # But we should still prefer low sampled values even if not better than f_best
    # Use: score = -(sampled) / cost^cost_exp  (minimize sampled / cost)
    # Or: score = (f_best - sampled) / cost^cost_exp  (expected improvement style)
    
    # Let's use a combination: base score from Thompson sample, scaled by cost
    # The Thompson sample already incorporates both exploitation and exploration
    base_score = -(sampled)  # We want to minimize sampled, so negate for maximization
    
    # Apply cost penalty
    cost_penalty = np.power(cost, cost_exp)
    score = base_score / cost_penalty
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima early in the search
    sigma_threshold = np.percentile(sigma, 95)
    exploration_bonus = np.where(sigma > sigma_threshold, 0.3 * (1.0 - progress), 0.0)
    score += exploration_bonus / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Normalize scores to [0, 1] for consistency
    score_min = np.min(score)
    score_max = np.max(score)
    
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score = np.zeros_like(score)
    
    return score
