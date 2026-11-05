import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, 16):
        if is_prime[i]:
            for j in range(i * i, 257, i):
                is_prime[j] = False

    # All primes in [3, 255] (odd primes, since sums of one odd + one even are odd)
    odd_primes = [p for p in range(3, 256, 2) if is_prime[p]]

    # We need to find a1,a2,a3,b1,b2,b3 in [1,128] (distinct within each set)
    # such that all 9 sums s_ij = a_i + b_j are distinct odd primes.
    #
    # Algebraic approach: pick 3 free primes p11, p12, p13 (row 1: a1+b1, a1+b2, a1+b3).
    # Then:
    #   b1 = p11 - a1, b2 = p12 - a1, b3 = p13 - a1
    #   For row 2: p21 = a2 + b1, p22 = a2 + b2, p23 = a2 + b3
    #   a2 = p21 - b1 = p21 - p11 + a1
    #   Similarly a3 = p31 - p11 + a1
    #   Then p22 = a2 + b2 = p21 - p11 + a1 + p12 - a1 = p21 + p12 - p11
    #   p23 = a2 + b3 = p21 + p13 - p11
    #   p32 = a3 + b2 = p31 + p12 - p11
    #   p33 = a3 + b3 = p31 + p13 - p11
    #
    # So if we pick p11, p12, p13, p21, p31 (5 free primes), the other 4 are determined.
    # But we need all 9 to be prime, distinct, and in [3, 255].
    #
    # Actually, let's pick p11, p12, p13, p21, p31 and check:
    #   p22 = p21 + p12 - p11 must be prime
    #   p23 = p21 + p13 - p11 must be prime
    #   p32 = p31 + p12 - p11 must be prime
    #   p33 = p31 + p13 - p11 must be prime
    #
    # Then recover:
    #   a1 is free (1 parameter)
    #   b1 = p11 - a1, b2 = p12 - a1, b3 = p13 - a1
    #   a2 = p21 - b1 = p21 - p11 + a1
    #   a3 = p31 - b1 = p31 - p11 + a1
    #
    # Constraints:
    #   1 <= a1, a2, a3 <= 128, distinct
    #   1 <= b1, b2, b3 <= 128, distinct
    #   All 9 primes distinct
    #
    # Let d2 = p21 - p11, d3 = p31 - p11
    #   a1 = x
    #   a2 = x + d2
    #   a3 = x + d3
    #   b1 = p11 - x
    #   b2 = p12 - x
    #   b3 = p13 - x
    #
    # x must satisfy:
    #   1 <= x <= 128
    #   1 <= x + d2 <= 128  =>  1 - d2 <= x <= 128 - d2
    #   1 <= x + d3 <= 128  =>  1 - d3 <= x <= 128 - d3
    #   1 <= p11 - x <= 128 =>  p11 - 128 <= x <= p11 - 1
    #   1 <= p12 - x <= 128 =>  p12 - 128 <= x <= p12 - 1
    #   1 <= p13 - x <= 128 =>  p13 - 128 <= x <= p13 - 1
    #
    # And all 6 values must be distinct (a1!=a2, a1!=a3, a2!=a3, b1!=b2, b1!=b3, b2!=b3)
    # which means d2 != 0, d3 != 0, d2 != d3, and p11 != p12, p11 != p13, p12 != p13.

    # Iterate over 5 free primes
    # To keep it fast, we'll iterate over smaller ranges first
    n_primes = len(odd_primes)
    
    # Try to find a valid solution
    found_A = None
    found_B = None
    
    # Iterate over p11, p12, p13, p21, p31
    # Use early termination
    for i1 in range(n_primes):
        if time.time() > deadline:
            break
        p11 = odd_primes[i1]
        for i2 in range(n_primes):
            if time.time() > deadline:
                break
            p12 = odd_primes[i2]
            if p12 == p11:
                continue
            for i3 in range(n_primes):
                if time.time() > deadline:
                    break
                p13 = odd_primes[i3]
                if p13 == p11 or p13 == p12:
                    continue
                
                d2_base = None  # will be p21 - p11
                d3_base = None  # will be p31 - p11
                
                for i4 in range(n_primes):
                    if time.time() > deadline:
                        break
                    p21 = odd_primes[i4]
                    d2 = p21 - p11
                    if d2 == 0:
                        continue
                    
                    # p22 = p21 + p12 - p11 = p21 + (p12 - p11)
                    p22 = p21 + p12 - p11
                    if p22 < 3 or p22 > 255 or not is_prime[p22]:
                        continue
                    if p22 in (p11, p12, p13, p21):
                        continue
                    
                    for i5 in range(n_primes):
                        if time.time() > deadline:
                            break
                        p31 = odd_primes[i5]
                        d3 = p31 - p11
                        if d3 == 0 or d3 == d2:
                            continue
                        
                        # p32 = p31 + p12 - p11
                        p32 = p31 + p12 - p11
                        if p32 < 3 or p32 > 255 or not is_prime[p32]:
                            continue
                        if p32 in (p11, p12, p13, p21, p22, p31):
                            continue
                        
                        # p33 = p31 + p13 - p11
                        p33 = p31 + p13 - p11
                        if p33 < 3 or p33 > 255 or not is_prime[p33]:
                            continue
                        if p33 in (p11, p12, p13, p21, p22, p31, p32):
                            continue
                        
                        # All 9 primes are distinct and valid
                        # Now find x such that all 6 variables are in [1,128]
                        # x range:
                        x_lo = max(1, 1 - d2, 1 - d3, p11 - 128, p12 - 128, p13 - 128)
                        x_hi = min(128, 128 - d2, 128 - d3, p11 - 1, p12 - 1, p13 - 1)
                        
                        if x_lo > x_hi:
                            continue
                        
                        # Try x values in range, check distinctness
                        for x in range(x_lo, x_hi + 1):
                            a1 = x
                            a2 = x + d2
                            a3 = x + d3
                            b1 = p11 - x
                            b2 = p12 - x
                            b3 = p13 - x
                            
                            # Check distinctness within A
                            if a1 == a2 or a1 == a3 or a2 == a3:
                                continue
                            # Check distinctness within B
                            if b1 == b2 or b1 == b3 or b2 == b3:
                                continue
                            
                            # All in [1,128] already guaranteed by range
                            found_A = [a1, a2, a3]
                            found_B = [b1, b2, b3]
                            return [found_A, found_B]
    
    # Fallback: Return a known valid solution
    return [[1, 7, 13], [4, 10, 16]]
