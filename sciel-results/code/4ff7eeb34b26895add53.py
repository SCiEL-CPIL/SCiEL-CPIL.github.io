"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction
import numpy as np

MAX_N = 10_000_000
TARGET = Fraction(3, 2)
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
    
    N = MAX_N
    
    # Step 1: Compute smallest prime factor (SPF) array up to N
    # This allows O(log n) sigma computation
    spf = np.zeros(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    
    for i in range(2, int(np.sqrt(N)) + 1):
        if spf[i] == 0:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    
    # Fill remaining primes
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
    
    if time_left() < 10:
        return [11, 6]
    
    # Step 2: Define function to compute sigma(n) using SPF
    def sigma(n):
        if n <= 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            # Compute p^k || n
            power = 1
            pk = 1
            while n % p == 0:
                n //= p
                pk *= p
                power += 1
            # sigma(p^k) = (p^(k+1) - 1) / (p - 1)
            # Use integer arithmetic
            result *= (pk * p - 1) // (p - 1)
        return result
    
    # Step 3: Search for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = 2a - 3b, so a = (3b + k) / 2
    # For integer a, need 3b + k even => b and k same parity
    # Error = |k| / (2b), so for fixed |k|, maximize b
    # Constraints: 1 <= a <= N, 1 <= b <= N, a != b
    
    best_err = None
    best_pair = (11, 6)
    
    # Check seed
    if sigma(11) == sigma(6):
        best_err = abs(Fraction(11, 6) - TARGET)
    
    # Iterate over small |k| values
    max_k = 50  # check |k| up to 50
    
    for k in range(1, max_k + 1):
        if time_left() < 3:
            break
        
        # Check both +k and -k
        for kk in [k, -k]:
            if time_left() < 2:
                break
            
            # a = (3b + kk) / 2 must be integer
            # 3b + kk even => b and kk same parity (since 3 is odd)
            
            # a <= N => 3b + kk <= 2N => b <= (2N - kk) / 3
            # a >= 1 => 3b + kk >= 2 => b >= (2 - kk) / 3
            # b >= 1
            
            b_max = (2 * N - kk) // 3
            b_min = max(1, (2 - kk + 2) // 3)
            
            if b_max < b_min:
                continue
            
            # Determine starting b based on parity
            if kk % 2 == 0:
                b_start = b_max if b_max % 2 == 0 else b_max - 1
            else:
                b_start = b_max if b_max % 2 == 1 else b_max - 1
            
            # Check b values from b_start downward
            # Limit checks to avoid timeout
            max_checks = 5000
            found = False
            
            for b in range(b_start, b_min - 1, -2):
                if time_left() < 2:
                    break
                if max_checks <= 0:
                    break
                max_checks -= 1
                
                a = (3 * b + kk) // 2
                if a < 1 or a > N or a == b:
                    continue
                
                if sigma(a) == sigma(b):
                    err = abs(Fraction(a, b) - TARGET)
                    if best_err is None or err < best_err:
                        best_err = err
                        best_pair = (a, b)
                    found = True
                    break
    
    # Verify result
    a_final, b_final = best_pair
    if sigma(a_final) != sigma(b_final):
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - TARGET)
    if err_final == 0:
        return [11, 6]
    
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
