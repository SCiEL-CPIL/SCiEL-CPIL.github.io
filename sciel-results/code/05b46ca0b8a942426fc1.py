import time
import math
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    N = 10_000_000

    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def sigma(n):
        if n <= 1:
            return n
        result = 1
        while n > 1:
            p = spf[n]
            power = 1
            sigma_p = 1
            while n % p == 0:
                n //= p
                power *= p
                sigma_p += power
            result *= sigma_p
        return result

    # Known good pair from prior work
    best_a, best_b = 122, 80
    best_num, best_den = 1, 40  # |122/80 - 3/2| = |122-120|/80 = 2/80 = 1/40

    # Meet-in-the-middle: generate candidate pairs near ratio 3/2
    # We want a/b ≈ 3/2, so a ≈ 3b/2
    # Try b from small to large, check a = floor(3b/2) and ceil(3b/2)
    # But first, let's be smart: only check b where 3b/2 is integer-ish
    
    # Strategy: For each b, the closest integers to 3b/2 are floor and ceil
    # We need sigma(a) = sigma(b) for a in {floor(3b/2), ceil(3b/2)}
    
    # To be more efficient, we can use the structure of sigma
    # sigma is multiplicative, so if a = 3b/2 exactly, we need special handling
    
    # Let's iterate b and check candidates
    # Use time budget wisely
    
    checked = 0
    max_checks = 2_000_000  # Limit total sigma computations
    
    for b in range(1, N + 1):
        if time.time() - start_time > deadline - 5:
            break
        if checked >= max_checks:
            break
            
        sb = sigma(b)
        checked += 1
        
        # Candidates for a: floor(3b/2) and ceil(3b/2)
        a1 = (3 * b) // 2
        a2 = a1 + 1 if (3 * b) % 2 != 0 else a1 + 1
        
        # Actually, 3b/2: if b is even, 3b/2 is integer; if odd, it's x.5
        if b % 2 == 0:
            # 3b/2 is integer
            a_target = 3 * b // 2
            candidates = [a_target]
            # Also check neighbors for safety
            if a_target > 1:
                candidates.append(a_target - 1)
            if a_target < N:
                candidates.append(a_target + 1)
        else:
            # 3b/2 = k + 0.5 where k = (3b-1)//2
            k = (3 * b - 1) // 2
            candidates = [k, k + 1]
            # Also check neighbors
            if k > 1:
                candidates.append(k - 1)
            if k + 1 < N:
                candidates.append(k + 2)
        
        for a in candidates:
            if a < 1 or a > N or a == b:
                continue
            
            sa = sigma(a)
            checked += 1
            
            if sa == sb:
                # Found a pair! Check the error
                # |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # Perfect match, but task says raw value of exactly 0 is invalid
                    # Actually, re-reading: "A raw value of exactly 0 cannot be scored and counts as invalid"
                    # So we skip perfect matches? Or does it mean a/b = 3/2 exactly?
                    # |a/b - 3/2| = 0 means a/b = 3/2, i.e., 2a = 3b
                    # If this happens, we need to find another pair
                    continue
                
                # Compare with best using exact rational arithmetic
                # num/den < best_num/best_den iff num*best_den < best_num*den
                if num * best_den < best_num * den:
                    best_num, best_den = num, den
                    best_a, best_b = a, b
    
    # If no better pair found, return the known good pair
    # But we should also do a local search around the best found
    # to see if we can improve further
    
    # Local refinement: check neighbors of best_a and best_b
    # Actually, let's do a focused search around ratios near 3/2
    # with a different approach: enumerate a and find b
    
    if time.time() - start_time < deadline - 10:
        # Try a different direction: for each a, find b near 2a/3
        for a in range(1, N + 1):
            if time.time() - start_time > deadline - 5:
                break
            if checked >= max_checks * 2:
                break
            
            sa = sigma(a)
            checked += 1
            
            # b ≈ 2a/3
            if a % 3 == 0:
                b_target = 2 * a // 3
                candidates = [b_target]
                if b_target > 1:
                    candidates.append(b_target - 1)
                if b_target < N:
                    candidates.append(b_target + 1)
            else:
                # 2a/3 is not integer
                k = (2 * a) // 3
                r = (2 * a) % 3
                if r == 0:
                    candidates = [k]
                elif r == 1:
                    # 2a/3 = k + 1/3, so k and k+1
                    candidates = [k, k + 1]
                else:
                    # 2a/3 = k + 2/3, so k and k+1
                    candidates = [k, k + 1]
            
            for b in candidates:
                if b < 1 or b > N or b == a:
                    continue
                
                sb = sigma(b)
                checked += 1
                
                if sb == sa:
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    
                    if num * best_den < best_num * den:
                        best_num, best_den = num, den
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
