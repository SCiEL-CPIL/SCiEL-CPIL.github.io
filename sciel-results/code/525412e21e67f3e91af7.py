import time
_start_time = time.time()
_TIMEOUT = 320.0 # Safety margin

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9
    
    # Refinement: Balance Heuristic with Strong Hash
    # The "Weight-layered" approach (prioritizing low weight) underperformed.
    # Reverting to a "Balance" heuristic (favoring vectors with symbol counts
    # close to uniform n/3) which is known to pack better in F_3^n.
    # We retain the high-quality 64-bit hash mixing for deterministic tie-breaking.
    
    c0 = 0
    c1 = 0
    c2 = 0
    h = 0x9e3779b97f4a7c15 # Initial seed for mixing
    
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        
        # Strong mixing for hash
        h ^= (v + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0x45d9f3b) & 0xFFFFFFFFFFFFFFFF
        h = (h ^ (h >> 16))
        
    ideal = n / 3.0
    
    # Penalty for deviation from uniform distribution of symbols.
    # Lower deviation is better.
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Normalize hash to [0, 1)
    hash_float = float(h) / (1 << 64)
    
    # Priority: Lower deviation is higher priority.
    # Scale deviation to dominate the hash part.
    # Max deviation is O(n^2). For n=10, max dev ~ 66.
    # We use a factor of 100 to ensure dev dominates hash [0,1).
    return -dev * 100.0 + hash_float
