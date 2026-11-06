import time
import hashlib

_start_time = time.time()
_TIME_LIMIT = 300.0

# Precompute weights for the Hamming weight of each coordinate
# We use a simple hash-based priority that breaks ties in a deterministic way
# but also incorporates structure to avoid arithmetic progressions.

def priority(vector, n):
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5:
        # Fallback: return a simple deterministic value
        # This ensures we finish quickly
        return -sum(vector)
    
    # Strategy: favor vectors with lower Hamming weight (fewer 1s/2s)
    # and use a hash to break ties in a way that correlates with cap-set-ness.
    # 
    # In Z_3^n, a 3-term AP {a, b, c} satisfies a + c = 2b (mod 3) for each coordinate.
    # This means for each coordinate i, one of the following must hold:
    #   a_i = c_i = b_i  (all same)
    #   a_i, c_i, b_i are all distinct (i.e., {0,1,2})
    # 
    # To minimize APs, we want to avoid sets where many vectors share coordinates 
    # in ways that create these patterns. 
    # 
    # A simple heuristic: prioritize vectors with "balanced" symbol counts 
    # (close to n/3 of each symbol) because extreme imbalances tend to create 
    # more APs in greedy constructions.
    
    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Measure deviation from balance
    balance_score = abs(c0 - n/3) + abs(c1 - n/3) + abs(c2 - n/3)
    
    # Use a hash for deterministic tie-breaking
    # Convert vector to bytes for hashing
    vec_bytes = bytes(vector)
    hash_val = int.from_bytes(hashlib.md5(vec_bytes).digest()[:8], 'big')
    
    # Combined priority: lower balance_score is better, then use hash
    # We want higher priority = earlier in greedy order
    # So invert balance_score
    priority_val = -balance_score + (hash_val / 2**63)
    
    return priority_val
