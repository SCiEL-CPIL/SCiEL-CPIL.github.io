"""Online bin packing: priority(item, bins) -> scores (higher is better)."""
import numpy as np

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity; the item goes to the highest-scoring bin.
    
    Strategy: Polarization with smooth surrogate.
    - Polarization: push bin remainders toward 0 or CAPACITY (avoid the middle).
    - Smooth surrogate: replace min(rem, cap-rem) with a soft-min using log-sum-exp
      to get a differentiable, smoother objective that still captures the same intent.
    
    Score = -soft_min(rem_after, CAPACITY - rem_after)
    where soft_min(a, b) = -log(exp(-a/t) + exp(-b/t)) * t  (for small t, approx min(a,b))
    
    We use t=2.0 for a moderate smoothing.
    """
    t = 2.0  # smoothing parameter; smaller = sharper (closer to true min)
    
    # Remaining capacity after placing the item
    rem_after = bins - item
    
    # Valid bins: those that can fit the item
    valid_mask = rem_after >= 0.0
    
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # For valid bins, compute soft_min(rem_after, CAPACITY - rem_after)
    # soft_min(a, b) = -t * log(exp(-a/t) + exp(-b/t))
    # To avoid overflow/underflow, use log-sum-exp trick:
    # Let u1 = rem_after / t, u2 = (CAPACITY - rem_after) / t
    # soft_min = -t * log(exp(-u1) + exp(-u2))
    #          = t * log(1 + exp(-(u1 - u2)))  ... wait, let me derive properly.
    
    # soft_min(a, b) = -t * log(exp(-a/t) + exp(-b/t))
    # Let m = min(a/t, b/t). Then:
    # soft_min = -t * [m + log(exp(-(a/t - m)) + exp(-(b/t - m)))]
    #          = -t * m - t * log(exp(-(a/t - m)) + exp(-(b/t - m)))
    
    # Since we're dealing with rem_after in [0, CAPACITY], and CAPACITY=100, t=2:
    # rem_after/t in [0, 50], (CAPACITY - rem_after)/t in [0, 50].
    # So u1, u2 are in [0, 50]. exp(-u) for u up to 50 is ~1.9e-22, which is fine for float64.
    
    a = rem_after
    b = CAPACITY - rem_after
    
    # For numerical stability, compute soft_min properly
    # soft_min(a, b) = -t * log(exp(-a/t) + exp(-b/t))
    u1 = a / t
    u2 = b / t
    
    # log(exp(-u1) + exp(-u2)) = max(-u1, -u2) + log(exp(-u1 - max(-u1,-u2)) + exp(-u2 - max(-u1,-u2)))
    m = np.maximum(u1, u2)  # max of the arguments (for log-sum-exp stability)
    # Actually we need: log(exp(-u1) + exp(-u2))
    # = max(-u1, -u2) + log(exp(-u1 - max(-u1,-u2)) + exp(-u2 - max(-u1,-u2)))
    # Let M = max(u1, u2). Then -u1 = -(M + (u1-M)) ... this is getting complicated.
    
    # Simpler: since u1, u2 <= 50, exp(-u1), exp(-u2) are at least ~1.9e-22, no underflow to 0.
    # Just compute directly.
    e1 = np.exp(-u1)
    e2 = np.exp(-u2)
    log_sum = np.log(e1 + e2)
    soft_min_val = -t * log_sum  # This is positive (since log_sum is negative or small)
    
    # Wait: if u1=0, u2=50, e1=1, e2~0, log_sum~0, soft_min~0. Correct (min(0,100)=0).
    # If u1=25, u2=25, e1=e2~1.4e-11, log_sum~log(2.8e-11)~-24.6, soft_min~-2*(-24.6)=49.2. 
    # min(50,50)=50. Close enough with t=2.
    
    scores = -soft_min_val  # We want to MINIMIZE soft_min, so score = -soft_min (higher is better)
    
    # Mask invalid bins
    scores = np.where(valid_mask, scores, -np.inf)
    
    return scores
