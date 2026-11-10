import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
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
        # Fallback if seed is 0 (it's not, but just in case)
        best_val = float('inf')
        best_coeffs = [0] * n

    # Local Search
    # We want to minimize |S| subject to S != 0
    
    # Helper to compute sum
    def calc_sum(coeffs):
        return sum(c * w[i] for i, c in enumerate(coeffs))
    
    # Random restart local search
    # Each restart:
    # 1. Start from a random config or best found
    # 2. Iterate: pick random index, try changing to -1, 0, 1
    # 3. If improvement (smaller non-zero abs sum), accept
    # 4. If stuck, break
    
    random.seed(42) # Deterministic for reproducibility in dev, but maybe random is better? 
    # Using a fixed seed ensures consistent behavior, but random might explore more. 
    # Let's use a mix or just random.
    
    max_iters_per_restart = 20000
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current state
        if restart == 0:
            curr = best_coeffs[:]
        else:
            # Random initialization
            curr = [random.choice([-1, 0, 1]) for _ in range(n)]
            
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        # If we started with inf, we must find a non-zero sum
        # If we started with a value, we try to improve
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try a few random moves
            for _ in range(max_iters_per_restart):
                if time.time() > deadline - 5.0:
                    break
                
                idx = random.randint(0, n - 1)
                old_c = curr[idx]
                
                # Try other values
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * w[idx]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    
                    # Accept if new_abs is better (smaller) than curr_abs
                    # Note: if curr_abs is inf (sum was 0), any finite new_abs is better
                    if new_abs < curr_abs:
                        curr[idx] = new_c
                        curr_sum = new_sum
                        curr_abs = new_abs
                        improved = True
                        
                        # Update global best
                        if curr_abs < best_val:
                            best_val = curr_abs
                            best_coeffs = curr[:]
                        break # Accept first improvement and restart inner loop? 
                             # Or continue searching for better in this step?
                             # Usually steepest descent or first improvement. 
                             # First improvement is faster.
            
        # If we found a better solution, we can restart from it? 
        # Or just continue.
        
    return best_coeffs
# EVOLVE-BLOCK-END
