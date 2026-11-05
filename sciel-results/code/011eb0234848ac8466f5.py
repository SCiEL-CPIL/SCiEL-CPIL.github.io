import time
from math import gcd
from itertools import combinations

N = 512
DEADLINE = 85.0

def is_compatible(x, S):
    for y in S:
        if (x // gcd(x, y)) * y > N:
            return False
    return True

def get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return divs

def construct():
    start_time = time.time()
    
    # Find M <= 512 with max divisors
    best_M = 1
    best_tau = 0
    for M in range(1, N + 1):
        tau = len(get_divisors(M))
        if tau > best_tau:
            best_tau = tau
            best_M = M
            
    # Start with divisors of best_M
    best_S = set(get_divisors(best_M))
    best_size = len(best_S)
    
    # Local Search: Try to increase size by swapping
    # Strategy: Remove k elements, add k+1 elements
    # We focus on k=1 (remove 1, add 2) and k=2 (remove 2, add 3)
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # Try remove 1, add 2
        for y in list(best_S):
            S_temp = best_S - {y}
            # Find candidates x1, x2 not in S_temp
            # To be efficient, we can just iterate x1, x2
            # But 512^2 is ~260k, which is fine.
            # We need x1, x2 compatible with S_temp and each other.
            
            # Optimization: Pre-compute compatibility with S_temp?
            # Just check on the fly.
            
            candidates = []
            for x in range(1, N + 1):
                if x not in S_temp and is_compatible(x, S_temp):
                    candidates.append(x)
            
            # Now find pair x1, x2 in candidates such that they are compatible
            # And we want to add both.
            # Actually, if we add x1, we must check if x2 is compatible with S_temp + {x1}.
            # It's easier to just check pairs.
            
            found_pair = False
            for i in range(len(candidates)):
                if time.time() > DEADLINE: break
                x1 = candidates[i]
                S_with_x1 = S_temp | {x1}
                for j in range(i + 1, len(candidates)):
                    x2 = candidates[j]
                    if is_compatible(x2, S_with_x1):
                        # We can add both
                        new_S = S_with_x1 | {x2}
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(best_S)
                            improved = True
                            found_pair = True
                            break
                if found_pair:
                    break
            if found_pair:
                break # Restart loop with new best_S

        if improved:
            continue

        # Try remove 2, add 3
        # This is more expensive. Only if time permits.
        if time.time() < DEADLINE - 10:
             for pair in combinations(list(best_S), 2):
                y1, y2 = pair
                S_temp = best_S - {y1, y2}
                candidates = []
                for x in range(1, N + 1):
                    if x not in S_temp and is_compatible(x, S_temp):
                        candidates.append(x)
                
                # Find triple x1, x2, x3
                # Heuristic: Just try to add 3 greedily?
                # Or check triples.
                # If candidates is large, this is slow.
                # Let's just try to add 3 greedily from candidates
                # and see if we can get size +1.
                
                S_new = S_temp.copy()
                added = 0
                for x in candidates:
                    if is_compatible(x, S_new):
                        S_new.add(x)
                        added += 1
                        if added == 3:
                            break
                
                if len(S_new) > best_size:
                    best_S = S_new
                    best_size = len(best_S)
                    improved = True
                    break

    return list(best_S)
