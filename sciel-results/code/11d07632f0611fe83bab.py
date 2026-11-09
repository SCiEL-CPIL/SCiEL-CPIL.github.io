import hashlib
import time

# We don't need time in priority, the evaluator handles the deadline.
# But we must be fast.

def priority(vector, n):
    # Fast counting using C-level methods
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = n - c0 - c1
    
    # Balance metric: Minimize sum of squares of counts.
    # This is equivalent to minimizing variance from n/3.
    # Lower sum of squares -> more balanced.
    # We want higher priority for balanced, so negate.
    balance_score = -(c0 * c0 + c1 * c1 + c2 * c2)
    
    # Diversity bonus: Reward vectors that contain all three symbols.
    # Vectors in hyperplanes (missing a symbol) might be less effective
    # for maximizing the global cap set size when combined with others,
    # or this heuristic simply favors "central" vectors in the cube.
    diversity_bonus = 1.0 if (c0 > 0 and c1 > 0 and c2 > 0) else 0.0
    
    # Jitter for tie-breaking
    # Use built-in hash for speed (implemented in C)
    h = hash(vector) & 0xFFFFFFFF
    jitter = (h % 1000) / 1000.0
    
    return balance_score + diversity_bonus + jitter
