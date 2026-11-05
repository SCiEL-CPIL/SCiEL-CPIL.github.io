"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Find the number M <= 512 with the maximum number of divisors
    best_m = 1
    max_divs = 0
    best_divs = [1]
    
    for m in range(1, 513):
        # Count divisors
        divs = []
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
            i += 1
        
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_m = m
            best_divs = divs
            
        if time.time() > deadline:
            break
            
    # Verify validity (should be true by construction)
    # And try to add any additional numbers if possible (greedy)
    S = set(best_divs)
    
    # Try to add numbers not in S
    # This might not yield improvements, but it's a safe check
    # We check if adding x keeps the property for all existing y
    # Since S is divisors of M, lcm(x, y) <= 512 for all y in S
    # is equivalent to lcm(x, M) <= 512? No, lcm(x, y) divides lcm(x, M) only if y divides M.
    # But lcm(x, M) is the max LCM with any divisor of M?
    # lcm(x, y) divides lcm(x, M). So if lcm(x, M) <= 512, then all lcm(x, y) <= 512.
    
    for x in range(1, 513):
        if x in S:
            continue
        # Check if x can be added
        # We need lcm(x, y) <= 512 for all y in S
        # It suffices to check lcm(x, best_m) <= 512?
        # No, lcm(x, y) <= lcm(x, best_m) is not always true in terms of value?
        # Actually lcm(x, y) divides lcm(x, best_m) because y divides best_m.
        # So lcm(x, best_m) is a multiple of lcm(x, y).
        # Thus if lcm(x, best_m) <= 512, then lcm(x, y) <= 512.
        
        import math
        g = math.gcd(x, best_m)
        lcm_val = (x // g) * best_m
        if lcm_val <= 512:
            S.add(x)
            
    if time.time() > deadline:
        return list(best_divs)
        
    return list(S)

# EVOLVE-BLOCK-END
