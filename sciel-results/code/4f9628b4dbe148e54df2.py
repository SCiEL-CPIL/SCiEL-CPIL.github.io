import math

def priority(vector, n):
    # Calculate counts of 0, 1, 2
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Ideal count per symbol
    ideal = n / 3.0
    
    # Deviation from ideal balance
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # We want to minimize deviation, so we negate it for "higher is better"
    # Scale it up to dominate the hash tie-breaker
    balance_score = -deviation * 1000.0
    
    # Deterministic hash for tie-breaking
    # Using a simple polynomial hash as before, but scaled down
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    
    # Combine: Balance is the primary driver, hash breaks ties
    # We map the hash to a small range [0, 1) to act as a tie-breaker
    hash_score = h / (10**9 + 7)
    
    return float(balance_score + hash_score)
