import time
import sys
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import bisect
MAX_N = 10_000_000
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
        
    N = MAX_N
    
    # Step 1: Compute sigma(n) for all n up to N using NumPy sieve
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for sigma
    for i in range(1, N + 1):
        if time_left() < 10:
            return [11, 6]
        sigma_np[i::i] += i
        
    if time_left() < 5:
        return [11, 6]
        
    best_err = float('inf')
    best_pair = (11, 6)
    
    # Step 2: Group indices by sigma value
    sigma_to_indices = {}
    for i in range(1, N + 1):
        if time_left() < 5:
            return [11, 6]
        s = int(sigma_np[i])
        if s in sigma_to_indices:
            sigma_to_indices[s].append(i)
        else:
            sigma_to_indices[s] = [i]
            
    if time_left() < 2:
        return [11, 6]
        
    # Sort the lists in the dictionary to enable binary search
    for s in sigma_to_indices:
        sigma_to_indices[s].sort()
        
    if time_left() < 2:
        return [11, 6]

    # Step 3: Search for best pair
    # Iterate b from N down to 1
    b = N
    while b >= 1:
        if time_left() < 2:
            break
            
        # Check pruning condition
        if best_err < float('inf'):
            # 1/(2b) >= best_err  =>  1 >= 2*b*best_err
            if 1.0 >= 2.0 * b * best_err:
                break
                
        s = int(sigma_np[b])
        candidates = sigma_to_indices.get(s, [])
        
        if not candidates:
            b -= 1
            continue
            
        # Target a is 1.5 * b
        target = 1.5 * b
        
        # Binary search for target in candidates
        pos = bisect.bisect_left(candidates, target)
        
        # Check neighbors
        checked = set()
        for offset in [-2, -1, 0, 1, 2]:
            idx = pos + offset
            if 0 <= idx < len(candidates):
                a = candidates[idx]
                if a in checked:
                    continue
                checked.add(a)
                
                if a == b:
                    continue
                    
                if 1 <= a <= N:
                    err = abs(a / b - 1.5)
                    
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
        b -= 1
        
    # Verify result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if a_final == b_final:
        return [11, 6]
    if sigma_np[a_final] != sigma_np[b_final]:
        return [11, 6]
        
    err_final = abs(Fraction(a_final, b_final) - Fraction(3, 2))
    if err_final == 0:
        return [11, 6]
        
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
