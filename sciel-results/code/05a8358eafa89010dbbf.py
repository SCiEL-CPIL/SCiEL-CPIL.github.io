"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
import numpy as np

MAX_N = 10_000_000
TARGET = Fraction(3, 2)
DEADLINE_SECONDS = 80  # safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
    
    N = MAX_N
    
    # Step 1: Compute sigma(n) for all n up to N using NumPy sieve
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        if time_left() < 15:  # reserve 15s for search
            break
        sigma_np[i::i] += i
    
    if time_left() < 5:
        return [11, 6]
    
    # Step 2: Search for best pair using error-numerator approach
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = 2a - 3b, so a = (3b + k) / 2
    # For k to give integer a, we need 3b + k even, i.e., b and k have same parity
    # Error = |k| / (2b), so for fixed k, we want large b
    # But we need a <= N, so (3b + k) / 2 <= N => b <= (2N - k) / 3
    
    best_err = None
    best_pair = (11, 6)
    
    # Check seed
    if sigma_np[11] == sigma_np[6]:
        best_err = abs(Fraction(11, 6) - TARGET)
    
    # Iterate over small |k| values
    # For each k, find largest b such that:
    #   1. b >= 1
    #   2. a = (3b + k) / 2 is integer (3b + k even)
    #   3. 1 <= a <= N
    #   4. a != b (to avoid ratio 1, though |a/b - 3/2| = 1/2 for a=b)
    #   5. sigma(a) == sigma(b)
    #   6. Minimize |k| / (2b), i.e., maximize b for fixed |k|
    
    max_k = 200  # check |k| up to 200
    
    for k in range(0, max_k + 1):
        if time_left() < 3:
            break
        
        # Check both +k and -k (except k=0)
        ks_to_check = [k] if k == 0 else [k, -k]
        
        for kk in ks_to_check:
            if time_left() < 2:
                break
            
            # For this kk, find valid b values
            # a = (3b + kk) / 2 must be integer => 3b + kk even
            # 3b + kk even means b and kk have same parity (since 3 is odd)
            
            # a <= N => 3b + kk <= 2N => b <= (2N - kk) / 3
            # a >= 1 => 3b + kk >= 2 => b >= (2 - kk) / 3
            # b >= 1
            
            # Start from largest valid b and work down until we find sigma match
            b_max = (2 * N - kk) // 3
            b_min = max(1, (2 - kk + 2) // 3)  # ensure b >= 1
            
            if b_max < b_min:
                continue
            
            # We need 3b + kk to be even
            # 3b + kk even => b has same parity as kk (since 3 is odd)
            # If kk is even, b must be even
            # If kk is odd, b must be odd
            
            if kk % 2 == 0:
                # b must be even
                b_start = b_max if b_max % 2 == 0 else b_max - 1
            else:
                # b must be odd
                b_start = b_max if b_max % 2 == 1 else b_max - 1
            
            # Check b values from b_start downward, but limit the search
            # For small |k|, the error |k|/(2b) decreases as b increases
            # So we want the largest b with sigma(a) == sigma(b)
            
            # But checking all b from b_start down could be slow
            # Limit to checking a reasonable number of candidates
            max_checks = 10000
            
            for b in range(b_start, b_min - 1, -2):  # step by 2 to maintain parity
                if time_left() < 2:
                    break
                if max_checks <= 0:
                    break
                max_checks -= 1
                
                a = (3 * b + kk) // 2
                if a < 1 or a > N or a == b:
                    continue
                
                if sigma_np[a] == sigma_np[b]:
                    err = abs(Fraction(a, b) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a, b)
                    # Found a match for this k; for larger b we'd have smaller error
                    # But we started from largest b, so this is the best for this k
                    break
    
    # Also do a vectorized search for exact 3b/2 case (k=0 or k=1)
    if time_left() > 5:
        b_arr = np.arange(1, N + 1, dtype=np.int64)
        a_target = (3 * b_arr) // 2
        
        valid = (a_target >= 1) & (a_target <= N) & (a_target != b_arr)
        a_target_valid = a_target[valid]
        b_valid = b_arr[valid]
        
        sigma_a = sigma_np[a_target_valid]
        sigma_b = sigma_np[b_valid]
        matches = sigma_a == sigma_b
        
        if np.any(matches):
            a_matches = a_target_valid[matches]
            b_matches = b_valid[matches]
            
            num = np.abs(2 * a_matches - 3 * b_matches)
            den = 2 * b_matches
            ratios = num.astype(np.float64) / den.astype(np.float64)
            min_idx = np.argmin(ratios)
            
            a_best = int(a_matches[min_idx])
            b_best = int(b_matches[min_idx])
            
            if a_best != b_best and a_best <= N and b_best <= N:
                err = abs(Fraction(a_best, b_best) - TARGET)
                if best_err is None or err < best_err:
                    best_err = err
                    best_pair = (a_best, b_best)
    
    # Verify result
    a_final, b_final = best_pair
    if sigma_np[a_final] != sigma_np[b_final]:
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        return [11, 6]
    
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
