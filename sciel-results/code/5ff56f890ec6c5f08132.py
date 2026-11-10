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
        
    # Step 2: Iterate k from 1 upward, find minimal error pair
    # Error = k / (2*b) for |2a - 3b| = k
    # For each k, try largest valid b first
    
    best_err = float('inf')
    best_pair = (11, 6)
    
    # Maximum k to check: even with b=1, error = k/2, so k up to some limit
    # We'll iterate k and stop when k/(2*1) > best_err (i.e., k > 2*best_err)
    # But initially best_err is inf, so we need a practical limit
    
    # For k, valid b ranges:
    # Case +: 2a - 3b = k => a = (3b+k)/2, need 3b+k even, 1<=a<=N
    #   b_max_plus = floor((2N - k) / 3) if (2N-k) >= 0 and parity works
    # Case -: 2a - 3b = -k => a = (3b-k)/2, need 3b-k even, 1<=a<=N  
    #   b_max_minus = floor((2N + k) / 3), b_min_minus = ceil((k+2)/3)
    
    max_k = 2000000  # practical limit; will be pruned by best_err
    
    for k in range(1, max_k + 1):
        if time_left() < 5:
            break
            
        # Pruning: minimum possible error for this k is k/(2*N)
        # If k/(2*N) >= best_err, we can stop
        if best_err < float('inf') and k / (2.0 * N) >= best_err:
            break
            
        # Case 1: 2a - 3b = k (positive)
        # a = (3b + k) / 2, need 3b + k even
        # 1 <= (3b+k)/2 <= N => 2-k <= 3b <= 2N-k => ceil((2-k)/3) <= b <= floor((2N-k)/3)
        # Also b >= 1
        if k <= 2 * N:  # need 2N-k >= 0
            b_lo = max(1, -(-((2 - k)) // 3))  # ceil((2-k)/3)
            b_hi = (2 * N - k) // 3  # floor((2N-k)/3)
            
            if b_lo <= b_hi:
                # Need 3b + k even => 3b and k have same parity => b and k have same parity (since 3 is odd)
                # So b % 2 == k % 2
                # Start from largest b with correct parity
                b_start = b_hi
                if b_start % 2 != k % 2:
                    b_start -= 1
                
                # Check a few b values near the top
                for _ in range(50):  # check up to 50 candidates
                    if time_left() < 3:
                        break
                    if b_start < b_lo:
                        break
                    
                    a = (3 * b_start + k) // 2
                    if 1 <= a <= N and a != b_start:
                        if sigma_np[a] == sigma_np[b_start]:
                            err = k / (2.0 * b_start)
                            if err < best_err:
                                best_err = err
                                best_pair = (a, b_start)
                                # Since we're iterating k in order, and for this k we found a match,
                                # we can potentially stop if this is the minimal possible
                                # But there might be better b for same k, so continue checking
                            # For this k, we want the largest b, so if we found one, 
                            # smaller b for same k give larger error, so we can break
                            break
                    
                    b_start -= 2  # maintain parity
                
        # Case 2: 2a - 3b = -k (negative)
        # a = (3b - k) / 2, need 3b - k even
        # 1 <= (3b-k)/2 <= N => 2+k <= 3b <= 2N+k => ceil((2+k)/3) <= b <= floor((2N+k)/3)
        # Also b >= 1
        b_lo = -(-(2 + k) // 3)  # ceil((2+k)/3)
        b_hi = (2 * N + k) // 3  # floor((2N+k)/3)
        
        if b_lo <= b_hi:
            # Need 3b - k even => 3b and k have same parity => b and k have same parity
            b_start = b_hi
            if b_start % 2 != k % 2:
                b_start -= 1
            
            # Check a few b values near the top
            for _ in range(50):
                if time_left() < 3:
                    break
                if b_start < b_lo:
                    break
                
                a = (3 * b_start - k) // 2
                if 1 <= a <= N and a != b_start:
                    if sigma_np[a] == sigma_np[b_start]:
                        err = k / (2.0 * b_start)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b_start)
                        break
                
                b_start -= 2  # maintain parity
    
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
