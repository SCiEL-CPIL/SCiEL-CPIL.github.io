"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for estimating fill probability of remainders
# These are tuned for typical Weibull sequences with capacity 100
# k=2.0, lambda=15.0 implies mean ~ 15 * Gamma(1.5) = 15 * 0.886 = 13.3
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0

# Pre-compute constants for speed
CAPACITY = 100.0
HALF_CAP = CAPACITY / 2.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins using a combined Polarization and Expected Dead Space heuristic.
    
    The strategy combines two structural insights:
    1. Polarization (from current program): Favors remainders close to 0 (full) 
       or close to Capacity (empty), avoiding "half-full" bins.
    2. Expected Dead Space (from inspiration 1): Models future items as Weibull 
       and penalizes remainders where the probability of being filled is low, 
       creating a bimodal preference that avoids "dead zone" medium-sized gaps.
    
    The combined score is:
    Score = (R - HALF_CAP)^2 - alpha * (r * exp(-(r/lambda)^k)) - beta * r
    
    Where:
    - (R - HALF_CAP)^2 is the polarization term (max at R=0 and R=CAPACITY)
    - r * exp(-(r/lambda)^k) is the dead space metric (max at intermediate r)
    - beta * r is a small consolidation bias to prevent pure Worst Fit
    
    This creates a score landscape that:
    - Strongly prefers very small remainders (tight packing)
    - Moderately prefers very large remainders (preserving flexibility)
    - Strongly penalizes medium remainders (dead zone)
    """
    global _START_TIME
    
    # Initialize timing if not started
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remainders = bins - item
    
    # Identify valid bins (enough capacity)
    valid_mask = remainders >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
        
    valid_rems = remainders[valid_mask]
    
    # Component 1: Polarization score
    # (R - HALF_CAP)^2 is max at R=0 and R=CAPACITY
    pol_scores = (valid_rems - HALF_CAP) ** 2
    
    # Component 2: Dead Space metric
    # r * S(r) where S(r) = exp(-(r/lambda)^k)
    # This has a maximum at intermediate r, penalizing "dead zone" remainders
    r = np.maximum(valid_rems, 0.0)
    ratio = r / _WEIBULL_LAMBDA
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    dead_space = r * survival
    
    # Component 3: Consolidation bias
    # Small penalty for large remainders to encourage filling bins
    beta = 0.01
    consolidation = beta * r
    
    # Combine scores
    # Higher score is better, so we subtract the penalties
    alpha = 0.1  # Weight for dead space penalty
    
    combined_scores = pol_scores - alpha * dead_space - consolidation
    
    scores[valid_mask] = combined_scores
    
    return scores
# EVOLVE-BLOCK-END
