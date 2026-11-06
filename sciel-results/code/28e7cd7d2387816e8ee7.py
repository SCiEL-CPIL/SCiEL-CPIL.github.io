import time
_start_time = time.time()
_TIMEOUT = 320.0 # Safety margin

def priority(vector, n):
    # Check timeout
    if time.time() - _start_time > _TIMEOUT:
        return -1e9

    c0 = 0
    c1 = 0
    c2 = 0
    h = 0
    
    # Single pass to count symbols and compute hash
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        h = (h * 31 + v) % 1000000007
            
    ideal = n / 3.0
    # Quadratic deviation from ideal balance
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Penalize vectors that are too "concentrated" in any single symbol
    # This helps avoid vectors that are likely to form arithmetic progressions
    max_count = max(c0, c1, c2)
    concentration_penalty = max_count * max_count / n if n > 0 else 0
    
    # Small bonus for having all three symbols present (more diverse)
    diversity_bonus = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity_bonus = 0.5
    
    # Combine: minimize dev and concentration, maximize diversity
    score = -dev - 0.1 * concentration_penalty + diversity_bonus - h * 1e-9
    
    return score
