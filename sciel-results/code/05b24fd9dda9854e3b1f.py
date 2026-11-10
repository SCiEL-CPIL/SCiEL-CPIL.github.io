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
    
    # Build SPF (smallest prime factor) array up to N
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, int(np.sqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Function to compute sigma(n) using SPF
    def sigma(n):
        if n == 1:
            return 1
        result = 1
        while n > 1:
            p = spf[n]
            power = 0
            while n % p == 0:
                n //= p
                power += 1
            s = 1
            for _ in range(power):
                s = s * p + 1
            result *= s
        return result
    
    # Fallback pair: sigma(11) = 12, sigma(6) = 12, error = 1/3
    best_pair = (11, 6)
    best_err = Fraction(1, 3)
    
    # Iterate over k = |2a - 3b| starting from 1
    k = 0
    while time_left() > 5:
        k += 1
        
        # Theoretical best error for this k: k / (2 * N)
        # If k / (2*N) >= best_err, stop
        if k * best_err.numerator >= 2 * N * best_err.denominator:
            break
        
        # Case 1: 2a - 3b = k, a = (3b + k)/2
        b_max1 = (2 * N - k) // 3
        b_min1 = max(1, (2 - k + 2) // 3)
        if b_max1 >= b_min1:
            if (b_max1 % 2) != (k % 2):
                b_max1 -= 1
            if (b_min1 % 2) != (k % 2):
                b_min1 += 1
            if b_max1 >= b_min1:
                for b in range(b_max1, max(b_max1 - 1000, b_min1 - 1), -1):
                    if time_left() < 5:
                        break
                    a = (3 * b + k) // 2
                    if a < 1 or a > N or a == b:
                        continue
                    if sigma(a) == sigma(b):
                        err = Fraction(k, 2 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                        break
        
        # Case 2: 2a - 3b = -k, a = (3b - k)/2
        b_max2 = (2 * N + k) // 3
        b_min2 = max(1, (k + 2 + 2) // 3)
        if b_max2 >= b_min2:
            if (b_max2 % 2) != (k % 2):
                b_max2 -= 1
            if (b_min2 % 2) != (k % 2):
                b_min2 += 1
            if b_max2 >= b_min2:
                for b in range(b_max2, max(b_max2 - 1000, b_min2 - 1), -1):
                    if time_left() < 5:
                        break
                    a = (3 * b - k) // 2
                    if a < 1 or a > N or a == b:
                        continue
                    if sigma(a) == sigma(b):
                        err = Fraction(k, 2 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                        break
    
    a_final, b_final = best_pair
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
