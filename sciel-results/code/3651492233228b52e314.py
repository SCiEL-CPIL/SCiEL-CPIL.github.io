"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to evaluate n in [1000, 1000000]
    # Window is n+1 to n+24
    # Max value to check is 1000000 + 24 = 1000024
    max_n = 1000000
    max_val = max_n + 24
    
    # Step 1: Identify 24-rough numbers (numbers with no prime factor < 24)
    # is_rough[x] = 1 if x has no small prime factors, 0 otherwise.
    # Note: 1 is technically 24-rough (no prime factors at all).
    # But our window is n+1..n+24 with n>=1000, so we only look at x >= 1001.
    
    is_rough = np.ones(max_val + 1, dtype=np.int8)
    is_rough[0] = 0 # 0 is not considered rough in this context, though irrelevant
    
    # For each small prime p, mark multiples as not rough
    for p in small_primes:
        # Mark all multiples of p as 0
        is_rough[p::p] = 0
        
    # Step 2: Compute the number of rough numbers in each window of size 24
    # We want to maximize the count of rough numbers in [n+1, n+24]
    # Let W[n] = sum(is_rough[n+1 : n+25])
    # We can use cumulative sum for O(1) window sums
    
    # cumsum[i] = sum(is_rough[0 : i])
    # sum(is_rough[a : b]) = cumsum[b] - cumsum[a]
    # Here a = n+1, b = n+25
    # W[n] = cumsum[n+25] - cumsum[n+1]
    
    cumsum = np.cumsum(is_rough)
    
    # We need W[n] for n in [1000, 1000000]
    # n+1 ranges from 1001 to 1000001
    # n+25 ranges from 1025 to 1000025
    
    # Create arrays for the indices
    # n_indices = np.arange(1000, 1000001)
    # start_idx = n_indices + 1
    # end_idx = n_indices + 25
    
    # To avoid creating huge temporary arrays if memory is a concern, 
    # but 1e6 integers is only 8MB, so it's fine.
    
    n_min = 1000
    n_max = 1000000
    
    start_idxs = np.arange(n_min + 1, n_max + 25)
    end_idxs = start_idxs + 24
    
    window_sums = cumsum[end_idxs] - cumsum[start_idxs]
    
    # Find the n with the maximum window sum
    # window_sums[i] corresponds to n = n_min + i
    max_idx = np.argmax(window_sums)
    best_n = n_min + max_idx
    max_rough_count = window_sums[max_idx]
    
    # The score is the number of distinct factors.
    # If max_rough_count > 0, distinct factors = (24 - max_rough_count) + 1
    # If max_rough_count == 0, distinct factors = 24
    
    # We just return best_n. The evaluator computes the score.
    
    return best_n
# EVOLVE-BLOCK-END
