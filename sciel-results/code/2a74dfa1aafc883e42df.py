import math
import time

_start_time = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

def priority(vector, n):
    # Count distinct symbols in the vector
    distinct = len(set(vector))
    # Count occurrences of each symbol (0, 1, 2)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # L1 deviation from uniform distribution (n/3 of each symbol)
    target = n / 3.0
    l1_dev = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    
    # Count value changes in the vector
    # More value changes = more "generic" = less likely to be part of a line
    changes = 0
    for i in range(1, n):
        if vector[i] != vector[i-1]:
            changes += 1
    
    # Entropy-based measure: higher entropy = more uniform = less structured
    entropy = 0.0
    if c0 > 0:
        p0 = c0 / n
        entropy -= p0 * math.log2(p0)
    if c1 > 0:
        p1 = c1 / n
        entropy -= p1 * math.log2(p1)
    if c2 > 0:
        p2 = c2 / n
        entropy -= p2 * math.log2(p2)
    
    # Priority: distinct count dominates, then balance, then geometric spread
    # Higher distinct count = more generic = less likely to form lines
    # Lower L1 deviation = more balanced = less structured
    # More changes = more "generic" pattern = less likely to form lines
    # Higher entropy = more uniform distribution = less structured
    return distinct * 1000.0 - l1_dev + changes * 0.5 + entropy * 2.0
