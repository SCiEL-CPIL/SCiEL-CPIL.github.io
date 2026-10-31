import time
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Fast check for deadline
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    # Fast symbol counting
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # Balance metric: Prefer vectors close to n/3 of each symbol
    # This is a proxy for "low saturation" or "high independence"
    ideal = n / 3.0
    balance = -(abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal))
    
    # Tie-breaker: Use a hash to break symmetries deterministically
    # This helps the greedy algorithm explore different regions of the space
    # A simple hash of the tuple
    h = hash(vector)
    
    # Combine: Balance is the primary driver, hash is a small perturbation
    # Scale hash to be small relative to balance differences
    # Balance range is roughly [-n, 0]. 
    # We want hash to not override balance, but break ties.
    # If balance is an integer or half-integer, hash can break ties.
    # Let's normalize hash to [0, 1) and add it.
    # Since balance is negative, we want to maximize it.
    # Let's add a small positive value based on hash.
    hash_val = (h % 1000000) / 1000000.0
    
    return balance + hash_val
