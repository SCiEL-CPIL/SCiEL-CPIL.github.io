import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    # c[i] corresponds to 1/(i+1)
    w = [L // (i + 1) for i in range(n)]
    
    best_val = None
    best_coeffs = None
    
    # Initial seed: [0]*28 + [1, -1]
    # Sum = 1/29 - 1/30 = 1/870
    # N_seed = L/870
    seed_coeffs = [0] * 28 + [1, -1]
    S_seed = sum(c * w[i] for i, c in enumerate(seed_coeffs))
    if S_seed != 0:
        best_val = abs(S_seed)
        best_coeffs = seed_coeffs[:]
    else:
        best_val = float('inf')
        best_coeffs = [0] * n

    # Helper to compute sum
    def calc_sum(coeffs):
        return sum(c * w[i] for i, c in enumerate(coeffs))
    
    random.seed(42)
    
    # Phase 1: Random restart local search with single-index moves
    max_iters_per_restart = 5000
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            curr = best_coeffs[:]
        else:
            curr = [random.choice([-1, 0, 1]) for _ in range(n)]
            
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for _ in range(max_iters_per_restart):
                if time.time() > deadline - 10.0:
                    break
                
                idx = random.randint(0, n - 1)
                old_c = curr[idx]
                
                # Try all 3 values, pick best improvement (steepest descent)
                best_new_c = None
                best_new_abs = curr_abs
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[idx]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    if new_abs < best_new_abs:
                        best_new_abs = new_abs
                        best_new_c = new_c
                
                if best_new_c is not None:
                    curr[idx] = best_new_c
                    curr_sum += (best_new_c - old_c) * w[idx]
                    curr_abs = best_new_abs
                    improved = True
                    if curr_abs < best_val:
                        best_val = curr_abs
                        best_coeffs = curr[:]
    
    # Phase 2: Pair-move local search on best solution
    # Try changing two indices simultaneously
    if time.time() < deadline - 2.0:
        curr = best_coeffs[:]
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        # Precompute for speed
        for _ in range(100000):
            if time.time() > deadline - 2.0:
                break
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            if i1 == i2:
                continue
            c1 = curr[i1]
            c2 = curr[i2]
            # Try all combinations of new values for both
            for nc1 in [-1, 0, 1]:
                if nc1 == c1:
                    continue
                for nc2 in [-1, 0, 1]:
                    if nc2 == c2:
                        continue
                    delta = (nc1 - c1) * w[i1] + (nc2 - c2) * w[i2]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    if new_abs < curr_abs:
                        curr[i1] = nc1
                        curr[i2] = nc2
                        curr_sum = new_sum
                        curr_abs = new_abs
                        if curr_abs < best_val:
                            best_val = curr_abs
                            best_coeffs = curr[:]
                        break
                else:
                    continue
                break
            else:
                continue
            break
    
    # Phase 3: Final polish - exhaustive single-index scan on best
    if time.time() < deadline - 0.5:
        curr = best_coeffs[:]
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        # Multiple passes of full scan
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
            improved = False
            for idx in range(n):
                old_c = curr[idx]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[idx]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    if new_abs < curr_abs:
                        curr[idx] = new_c
                        curr_sum = new_sum
                        curr_abs = new_abs
                        improved = True
                        break
            if not improved:
                break
        
        if curr_abs < best_val:
            best_val = curr_abs
            best_coeffs = curr[:]
    
    return best_coeffs
# EVOLVE-BLOCK-END
