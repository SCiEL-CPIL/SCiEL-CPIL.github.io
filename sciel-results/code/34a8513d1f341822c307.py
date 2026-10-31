import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 15.0
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

CAPACITY = 100.0
# Golden Ratio targets for remaining capacity
PHI = (1 + np.sqrt(5)) / 2.0
TARGET_1 = (1.0 / PHI) * CAPACITY  # ~38.2
TARGET_2 = (1.0 - 1.0 / PHI) * CAPACITY # ~61.8

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    remaining_after = bins - item
    feasible = remaining_after >= 0

    if not np.any(feasible):
        return np.full(len(bins), -np.inf)

    scores = np.full(len(bins), -np.inf)
    
    rem = remaining_after[feasible]
    
    # Smooth surrogate for distance to targets
    # We want to maximize the score, so we want to minimize distance.
    # Score = - (smooth distance)
    # Using Gaussian-like penalty: exp(-d^2 / 2 * sigma^2)
    # This is smooth and differentiable.
    # We want high score for small d.
    
    sigma = 5.0 # Width of the preference peak
    
    d1 = np.abs(rem - TARGET_1)
    d2 = np.abs(rem - TARGET_2)
    
    # Soft min of d1 and d2
    # min(d1, d2) approx
    # Use log-sum-exp for min? 
    # min(a, b) = -max(-a, -b)
    # max(-a, -b) = log(exp(-a) + exp(-b))
    # So min(a, b) = -log(exp(-a) + exp(-b))
    # But we want a smooth version that is 0 when a=b=0.
    # Actually, just take the minimum of the two Gaussian scores?
    # Score = max( exp(-d1^2/2s^2), exp(-d2^2/2s^2) )
    # This is smooth (max of smooth functions is smooth if we use soft-max, but hard max is fine for argmax mostly, though non-diff).
    # Let's use soft-max for the score.
    
    s1 = np.exp(-(d1**2) / (2 * sigma**2))
    s2 = np.exp(-(d2**2) / (2 * sigma**2))
    
    # Soft max of s1, s2
    # max(s1, s2) approx
    # We want the score to be high if EITHER target is close.
    # score = s1 + s2? No, that double counts if both are close (impossible as targets are 23.6 apart, sigma=5, so they don't overlap much).
    # If they don't overlap, max(s1, s2) is fine.
    # To be smooth:
    # logaddexp(log(s1), log(s2)) ? No, s1, s2 are in (0,1].
    # Let's just use np.maximum(s1, s2). It's non-smooth but the kink is at d1=d2, which is r=50.
    # At r=50, d1=d2=11.8. s1=s2=exp(-11.8^2/50) = exp(-2.78) = 0.06.
    # The kink is at a low score value. It likely doesn't affect argmax unless two bins are exactly at 50.
    
    scores[feasible] = np.maximum(s1, s2)
    
    return scores
