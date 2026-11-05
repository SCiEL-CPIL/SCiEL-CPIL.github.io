import time
import math
import random
import itertools

def construct():
    start = time.time()
    DEADLINE = start + 85  # Safety margin before 90s limit

    # Precompute primes up to 256
    _is_prime_arr = [True] * 257
    _is_prime_arr[0] = False
    _is_prime_arr[1] = False
    for i in range(2, int(256**0.5) + 1):
        if _is_prime_arr[i]:
            for j in range(i*i, 257, i):
                _is_prime_arr[j] = False

    primes = [i for i in range(2, 257) if _is_prime_arr[i]]

    # Strategy: Find A=[a1,a2,a3], B=[b1,b2,b3] such that all 9 sums are distinct primes.
    # Let A = {x, x+d2, x+d3} and B = {p1-x, p2-x, p3-x}.
    # Sums:
    # S11 = p1
    # S12 = p2
    # S13 = p3
    # S21 = x+d2 + p1-x = p1 + d2
    # S22 = p2 + d2
    # S23 = p3 + d2
    # S31 = p1 + d3
    # S32 = p2 + d3
    # S33 = p3 + d3
    #
    # We need p1, p2, p3, p1+d2, p2+d2, p3+d2, p1+d3, p2+d3, p3+d3 to be distinct primes.
    # Also 1 <= x, x+d2, x+d3 <= 128
    # And 1 <= p1-x, p2-x, p3-x <= 128.
    # x+d2 = p4 - p1 (where p4 is the sum S21)
    # x+d3 = p7 - p1 (where p7 is the sum S31)
    # Let p4 = p1+d2, p5=p2+d2, p6=p3+d2, p7=p1+d3, p8=p2+d3, p9=p3+d3.
    # d2 = p4 - p1
    # d3 = p7 - p1
    # x = a1.
    # b1 = p1 - x
    # b2 = p2 - x
    # b3 = p3 - x
    # a2 = x + d2 = x + p4 - p1
    # a3 = x + d3 = x + p7 - p1
    
    # Iterate over 5 base primes p1, p2, p3, p4, p7.
    # Derive d2 = p4-p1, d3 = p7-p1.
    # Check if p2+d2, p3+d2, p2+d3, p3+d3 are primes.
    # Check distinctness of all 9.
    # Find x.
    
    rng = random.Random(42)
    
    # We can also try structured search for faster convergence if random is slow, 
    # but 5000 samples of 5 primes from ~50 is small (50 choose 5 = 2M). 
    # Wait, 50 primes under 256? 
    # Primes under 256: 2, 3, 5, ..., 251. Count is about 50.
    # 50C5 is ~ 2 million. We only do 5000 random samples. That's very sparse.
    # Let's try increasing the sample size or using a deterministic scan of a subset.
    # Actually, 5000 samples might be enough if the probability is high enough.
    # Probability that 4 derived numbers are prime: (1/log(p))^4.
    # If p~100, log~4.6. (1/4.6)^4 ~ 1/450.
    # So ~1% of samples should work. 5000 * 0.01 = 50 candidates.
    
    for _ in range(20000):
        if time.time() >= DEADLINE:
            break
            
        # Sample 5 distinct primes
        p1, p2, p3, p4, p7 = rng.sample(primes, 5)
        
        d2 = p4 - p1
        d3 = p7 - p1
        
        # Derived sums
        p5 = p2 + d2  # S22
        p6 = p3 + d2  # S23
        p8 = p2 + d3  # S32
        p9 = p3 + d3  # S33
        
        # Check bounds and primality for derived
        if p5 < 2 or p5 > 256 or p6 < 2 or p6 > 256 or p8 < 2 or p8 > 256 or p9 < 2 or p9 > 256:
            continue
        if not _is_prime_arr[p5] or not _is_prime_arr[p6] or not _is_prime_arr[p8] or not _is_prime_arr[p9]:
            continue
            
        # Check distinctness of the 9 primes
        ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        if len(set(ps)) != 9:
            continue
            
        # Find x
        # Constraints:
        # 1 <= x <= 128
        # 1 <= x + d2 <= 128  =>  1 - d2 <= x <= 128 - d2
        # 1 <= x + d3 <= 128  =>  1 - d3 <= x <= 128 - d3
        # 1 <= p1 - x <= 128  =>  p1 - 128 <= x <= p1 - 1
        # 1 <= p2 - x <= 128  =>  p2 - 128 <= x <= p2 - 1
        # 1 <= p3 - x <= 128  =>  p3 - 128 <= x <= p3 - 1
        
        low = max(1, 
                  1 - d2, 
                  1 - d3, 
                  p1 - 128, 
                  p2 - 128, 
                  p3 - 128)
                  
        high = min(128, 
                   128 - d2, 
                   128 - d3, 
                   p1 - 1, 
                   p2 - 1, 
                   p3 - 1)
                   
        if low > high:
            continue
            
        # Scan x for distinctness of A and B
        # A = [x, x+d2, x+d3]
        # B = [p1-x, p2-x, p3-x]
        # We need A to have 3 distinct values and B to have 3 distinct values.
        # Also all 6 values (A union B) don't need to be distinct from each other, 
        # only within A and within B. 
        # Wait, the problem says "each containing exactly 3 distinct integers". 
        # It does NOT say A and B must be disjoint. 
        # However, the sums must be distinct. 
        # If A and B share an element, it might not cause duplicate sums immediately, 
        # but let's just check A distinct and B distinct.
        
        found_x = None
        for x in range(low, high + 1):
            a1 = x
            a2 = x + d2
            a3 = x + d3
            b1 = p1 - x
            b2 = p2 - x
            b3 = p3 - x
            
            A = [a1, a2, a3]
            B = [b1, b2, b3]
            
            if len(set(A)) == 3 and len(set(B)) == 3:
                # Verify sums are in range and distinct (we already checked primality/distinctness of values, 
                # but let's double check the actual sum calculation logic matches)
                # The sums are exactly the primes we checked. 
                # Just need to ensure no out of bounds.
                valid = True
                for a in A:
                    for b in B:
                        s = a + b
                        if s < 1 or s > 256:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found_x = x
                    break
                    
        if found_x is not None:
            # Final verification
            A = [found_x, found_x + d2, found_x + d3]
            B = [p1 - found_x, p2 - found_x, p3 - found_x]
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a+b)
            
            sum_list = list(sums)
            # Check pairwise coprime
            ok = True
            for i in range(len(sum_list)):
                for j in range(i+1, len(sum_list)):
                    if math.gcd(sum_list[i], sum_list[j]) != 1:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok and len(sums) == 9:
                return [A, B]
                
    # Fallback: Systematic search if random failed
    # Try small primes to increase chance of fitting in [1,128]
    # Or just return a known good solution if we found one previously? 
    # We don't store it. Let's do a quick deterministic scan of the first 20 primes.
    
    primes_small = primes[:20]
    for p1 in primes_small:
        if time.time() >= DEADLINE:
            break
        for p2 in primes_small:
            if p1 == p2: continue
            for p3 in primes_small:
                if p1 == p3 or p2 == p3: continue
                for p4 in primes_small:
                    if p4 in (p1,p2,p3): continue
                    for p7 in primes_small:
                        if p7 in (p1,p2,p3,p4): continue
                        d2 = p4 - p1
                        d3 = p7 - p1
                        p5 = p2 + d2
                        p6 = p3 + d2
                        p8 = p2 + d3
                        p9 = p3 + d3
                        
                        if p5 < 2 or p5 > 256 or p6 < 2 or p6 > 256 or p8 < 2 or p8 > 256 or p9 < 2 or p9 > 256:
                            continue
                        if not _is_prime_arr[p5] or not _is_prime_arr[p6] or not _is_prime_arr[p8] or not _is_prime_arr[p9]:
                            continue
                            
                        ps = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
                        if len(set(ps)) != 9:
                            continue
                            
                        low = max(1, 1-d2, 1-d3, p1-128, p2-128, p3-128)
                        high = min(128, 128-d2, 128-d3, p1-1, p2-1, p3-1)
                        
                        if low > high:
                            continue
                            
                        for x in range(low, high+1):
                            A = [x, x+d2, x+d3]
                            B = [p1-x, p2-x, p3-x]
                            if len(set(A)) < 3 or len(set(B)) < 3:
                                continue
                            
                            sums = set(a+b for a in A for b in B)
                            if len(sums) < 9:
                                continue
                                
                            sum_list = list(sums)
                            ok = all(math.gcd(sum_list[i], sum_list[j]) == 1 
                                     for i in range(len(sum_list)) 
                                     for j in range(i+1, len(sum_list)))
                            if ok:
                                return [A, B]
                                
    # Absolute fallback
    # This might not have 9 distinct sums, but it's a valid return.
    # Actually, the task requires maximizing cardinality. 
    # If we can't find 9, we should find the next best.
    # But the current best is 9. 
    # Returning a valid pair with fewer sums is better than an error.
    # Let's return a simple valid pair.
    # A=[1, 3, 5], B=[2, 4, 6] -> Sums: 3,5,7, 5,7,9, 7,9,11. 
    # Distinct: 3,5,7,9,11. 9 and 3 are not coprime. Invalid.
    # A=[1, 3, 7], B=[2, 4, 8] -> Sums: 3,5,9, 5,7,11, 9,11,15.
    # Distinct: 3,5,7,9,11,15. 9,3 bad. 15,3,5 bad.
    
    # Let's try to find a valid pair with 5 coprime sums in the fallback.
    rng = random.Random(99)
    best = None
    best_count = 0
    for _ in range(5000):
        if time.time() >= DEADLINE - 1:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        sums = set(a+b for a in A for b in B)
        sum_list = list(sums)
        # Check if all pairwise coprime? No, we want max count of a subset? 
        # No, "All DISTINCT values in A+B must be pairwise coprime."
        # So we must find A,B such that ALL distinct sums are pairwise coprime.
        ok = True
        for i in range(len(sum_list)):
            for j in range(i+1, len(sum_list)):
                if math.gcd(sum_list[i], sum_list[j]) != 1:
                    ok = False
                    break
            if not ok: break
        if ok:
            if len(sums) > best_count:
                best_count = len(sums)
                best = [A, B]
                
    if best:
        return best
        
    # Last resort
    return [[1, 2, 3], [4, 5, 6]]
