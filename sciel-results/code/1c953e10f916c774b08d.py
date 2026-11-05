"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit

    # Try to find a minimal solution (3 or 4 terms)
    # 3 terms: 1/a + 1/b + 1/c = 1
    # (a-1)(b-1)(c-1) ... complex, but we proved 3 terms impossible for d>=5 in small range.
    # Let's assume 4 terms is the minimum.
    
    # Search for 4 terms: 1/a + 1/b + 1/c + 1/d = 1
    # a < b < c < d
    # a >= 5
    
    max_a = 200
    max_b = 1000
    
    for a in range(5, max_a):
        if time.time() > deadline:
            break
        for b in range(a + 1, max_b):
            if time.time() > deadline:
                break
            
            # S = 1 - 1/a - 1/b
            # N/D = (ab - a - b) / ab
            N = a * b - a - b
            D = a * b
            
            if N <= 0:
                break
            
            # We need (Nc - D)(Nd - D) = D^2
            # Let X = Nc - D, Y = Nd - D
            # X * Y = D^2
            # c = (X + D) / N
            # d = (Y + D) / N
            
            D_sq = D * D
            
            # Iterate divisors of D_sq
            # Optimization: only check X up to D
            limit = int(math.isqrt(D_sq))
            
            found = False
            # We want c > b, so Nc - D > Nb - D => X > N*b - D
            min_X = N * b - D
            
            for X in range(max(1, min_X), limit + 1):
                if D_sq % X != 0:
                    continue
                
                Y = D_sq // X
                
                # c = (X + D) / N
                num_c = X + D
                if num_c % N != 0:
                    continue
                c = num_c // N
                
                # d = (Y + D) / N
                num_d = Y + D
                if num_d % N != 0:
                    continue
                d = num_d // N
                
                # Check distinctness and order
                if c > b and d > c:
                    return [a, b, c, d]
                elif c > d and c > b: # Just in case order swapped
                    return [a, b, d, c]

    # Fallback to DP if minimal search fails or times out
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
