"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys
from fractions import Fraction
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    N = 30
    
    # Compute LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    weights = [L // i for i in range(1, N + 1)]
    
    # Sort indices by weight descending for better pruning
    # We need to keep track of original indices to reconstruct coefficients
    sorted_indices = sorted(range(N), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in sorted_indices]
    
    # Precompute suffix sums of sorted_weights
    # suffix_sum[k] = sum of sorted_weights[k:]
    suffix_sum = [0] * (N + 1)
    for k in range(N - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + sorted_weights[k]
        
    best_abs = float('inf')
    best_coeffs = [0] * N
    
    # Initialize with a simple seed: [0]*28 + [1, -1] at indices 28, 29 (0-indexed)
    # Sum = 1/29 - 1/30 = 1/870
    seed_coeffs = [0] * N
    seed_coeffs[28] = 1
    seed_coeffs[29] = -1
    seed_sum = 0
    for i in range(N):
        if seed_coeffs[i] != 0:
            seed_sum += seed_coeffs[i] * (L // (i + 1))
    best_abs = abs(seed_sum)
    best_coeffs = seed_coeffs[:]
    
    # Also try a few random seeds to tighten bound
    for _ in range(1000):
        c = [random.choice([-1, 0, 1]) for _ in range(N)]
        s = 0
        for i in range(N):
            if c[i] != 0:
                s += c[i] * (L // (i + 1))
        if s != 0 and abs(s) < best_abs:
            best_abs = abs(s)
            best_coeffs = c[:]
            
    def dfs(idx, current_sum, coeffs):
        nonlocal best_abs, best_coeffs
        
        if idx == N:
            if current_sum != 0 and abs(current_sum) < best_abs:
                best_abs = abs(current_sum)
                best_coeffs = coeffs[:]
            return
            
        # Pruning
        remaining_max = suffix_sum[idx]
        # The final sum will be in [current_sum - remaining_max, current_sum + remaining_max]
        # Min abs value in this interval:
        if current_sum > remaining_max:
            min_possible_abs = current_sum - remaining_max
        elif current_sum < -remaining_max:
            min_possible_abs = -current_sum - remaining_max
        else:
            min_possible_abs = 0
            
        if min_possible_abs >= best_abs:
            return
            
        w = sorted_weights[idx]
        orig_idx = sorted_indices[idx]
        
        # Determine order of trying coefficients to find good solutions faster
        # If current_sum > 0, we prefer -1 to reduce sum
        # If current_sum < 0, we prefer 1 to increase sum
        # If current_sum == 0, try 0 first? Or 1/-1?
        
        choices = [-1, 0, 1]
        if current_sum > 0:
            choices = [-1, 1, 0]
        elif current_sum < 0:
            choices = [1, -1, 0]
        else:
            choices = [0, 1, -1]
            
        for c in choices:
            new_sum = current_sum + c * w
            coeffs[orig_idx] = c
            dfs(idx + 1, new_sum, coeffs)
            coeffs[orig_idx] = 0
            
        # Check time
        if time.time() > deadline:
            raise TimeoutError()

    try:
        dfs(0, 0, [0] * N)
    except TimeoutError:
        pass
        
    # Verify best_coeffs
    final_sum = 0
    for i in range(N):
        if best_coeffs[i] != 0:
            final_sum += best_coeffs[i] * (L // (i + 1))
            
    if final_sum == 0:
        # Fallback to seed
        return [0]*28 + [1, -1]
        
    return best_coeffs
# EVOLVE-BLOCK-END
