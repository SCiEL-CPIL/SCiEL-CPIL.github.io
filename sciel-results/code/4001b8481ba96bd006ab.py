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

# State for the meta-optimization
# We optimize (kappa, alpha) where score = (mu + kappa*sigma - f_best) / cost**alpha
# Note: Standard UCB for minimization is usually mu - kappa*sigma, but we want to MAXIMIZE score.
# So we define a "pseudo-improvement" score.
# Let's stick to a robust hybrid: 
# Score = (f_best - mu + kappa * sigma) / cost**alpha
# This is effectively EI-like (if kappa handles the non-linear part) or UCB-like.
# Actually, let's use a simpler, more interpretable form:
# Score = (f_best - mu) / cost**alpha + beta * sigma / cost**alpha
# This is (f_best - mu + beta * sigma) / cost**alpha.
# We optimize beta and alpha.

_KAPPA = 2.0  # Initial exploration coefficient
_ALPHA = 1.0 # Initial cost exponent

# We will use a simple random search with memory or a lightweight GP on the hyperparameters.
# Since we only have 2000 points and need to be fast, we can't run a heavy GP on hyperparams every step.
# Instead, we will use a "bandit" approach or a simple grid refinement based on the current best.
# However, the directive asks for a STRUCTURAL change. 
# Let's use a **Thompson Sampling** approach over a small set of candidate (kappa, alpha) pairs.
# We maintain a history of (kappa, alpha) and the resulting "regret" or "improvement" observed.
# But we don't know the true improvement until the next step.
# Alternative: Use the GP posterior itself to infer the optimal acquisition.
# 
# New Strategy: **Adaptive Cost-Weighted UCB with Online Hyperparameter Tuning**
# We maintain a small set of "experts" (kappa, alpha) pairs.
# Each step, we evaluate the acquisition score for all experts? No, that's too slow.
# 
# Let's try a different structural change: **Lookahead / Multi-Step Planning**
# Instead of greedy 1-step acquisition, we estimate the expected improvement of the TOP 3 candidates
# and pick the one that maximizes the *expected minimum* of the future 3 evaluations?
# This is computationally expensive.
#
# Let's go with **Hyperparameter Optimization via Simplex Search on the Acquisition Parameters**.
# We treat (kappa, alpha) as parameters. We keep a history of the last few acquisitions and the resulting f_best changes.
# If f_best improved significantly, we stay close to current params. If not, we explore.
# This is a meta-learning strategy.

_meta_history = [] # List of (kappa, alpha, f_best_before, f_best_after)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if _time_exceeded():
        # Fallback: standard EI / cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # --- Meta-Learning Logic ---
    # We need to update our knowledge of the best (kappa, alpha).
    # We don't have access to the previous f_best in the function signature directly 
    # (f_best is the current lowest observed). 
    # We can infer "did we improve?" by comparing current f_best to a stored global min?
    # No, f_best IS the global min so far.
    # We can store the previous f_best in the module scope.
    
    global _KAPPA, _ALPHA
    global _PREV_F_BEST
    
    # Check if we have a previous f_best to compare against
    if '_PREV_F_BEST' not in globals():
        _PREV_F_BEST = f_best
        # Initialize params randomly or with defaults
        _KAPPA = 2.0
        _ALPHA = 1.0
    else:
        # Update logic
        # If we improved, we likely had good params.
        # If we didn't improve, we might need to explore more (higher kappa) or change cost weight.
        improvement = _PREV_F_BEST - f_best
        
        # Simple heuristic update:
        # If no improvement, increase exploration (kappa) slightly.
        # If improvement, decrease exploration slightly to exploit.
        if improvement > 1e-5:
            _KAPPA = max(0.5, _KAPPA * 0.95)
            _ALPHA = max(0.5, _ALPHA * 0.95)
        else:
            _KAPPA = min(5.0, _KAPPA * 1.1)
            _ALPHA = min(2.0, _ALPHA * 1.1)
            
        _PREV_F_BEST = f_best

    # --- Compute Acquisition Score ---
    # Score = (f_best - mu + kappa * sigma) / cost**alpha
    # This is a cost-aware UCB (Upper Confidence Bound) adapted for minimization.
    # We want to maximize the potential drop in f.
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate the numerator: Expected Improvement proxy
    # Using UCB form: (f_best - mu) + kappa * sigma
    # This is linear in sigma, which is different from EI's non-linear behavior.
    # UCB is often better for multi-modal functions like Ackley/Rastrigin.
    
    numerator = (f_best - mu) + _KAPPA * sigma
    
    # Apply cost penalty
    denominator = cost_safe ** _ALPHA
    
    score = numerator / denominator
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
