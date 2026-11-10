import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    MAX_N = 10000000
    
    # Precompute sigma(n) for all n up to MAX_N using a linear sieve
    # This allows O(1) sigma lookups during the search phase.
    
    sigma = [0] * (MAX_N + 1)
    spow = [0] * (MAX_N + 1)  # p^k where p is the smallest prime factor
    sval = [0] * (MAX_N + 1)  # 1 + p + ... + p^k
    is_comp = [False] * (MAX_N + 1)
    primes = []
    
    sigma[1] = 1
    spow[1] = 1
    sval[1] = 1
    
    for i in range(2, MAX_N + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            sval[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            j = i * p
            if j > MAX_N:
                break
            is_comp[j] = True
            if i % p == 0:
                spow[j] = spow[i] * p
                sval[j] = sval[i] + spow[j]
                sigma[j] = (sigma[i] // sval[i]) * sval[j]
                break
            else:
                spow[j] = p
                sval[j] = 1 + p
                sigma[j] = sigma[i] * (1 + p)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Now search for pairs (a, b) with sigma(a) == sigma(b) minimizing |a/b - 3/2|
    # Strategy: iterate over small error numerators k = |2a - 3b|
    # For each k, solve 2a - 3b = k and 2a - 3b = -k
    # We want a, b in [1, MAX_N] and a != b (actually a/b != 3/2 exactly)
    # and sigma(a) == sigma(b)
    
    # For 2a - 3b = k:
    #   a = (3b + k) / 2, so 3b + k must be even, i.e., b and k have same parity
    #   1 <= a <= MAX_N => 1 <= (3b+k)/2 <= MAX_N => 2-k <= 3b <= 2*MAX_N - k
    #   1 <= b <= MAX_N
    #   So b ranges: max(1, ceil((2-k)/3)) to min(MAX_N, floor((2*MAX_N - k)/3))
    
    # For 2a - 3b = -k (i.e., 3b - 2a = k):
    #   b = (2a + k) / 3, so 2a + k must be divisible by 3
    #   1 <= b <= MAX_N => 3 <= 2a+k <= 3*MAX_N => (3-k)/2 <= a <= (3*MAX_N - k)/2
    #   1 <= a <= MAX_N
    
    # We'll iterate k from 1 upward. For each k, we check valid (a,b) pairs.
    # To be efficient, for each k we can iterate over valid b values and check sigma.
    # But for large k, there are many b values. However, we stop at the first valid pair.
    
    best_a, best_b = 11, 6
    best_err_num = abs(2*11 - 3*6)  # = |22-18| = 4
    # error = best_err_num / (2*b) ... actually |a/b - 3/2| = |2a-3b|/(2b)
    
    # Let's just track the actual error as a rational to compare
    # error = |2a - 3b| / (2*b)
    
    # We'll search k = 1, 2, 3, ...
    # For each k, find valid (a, b) with sigma(a) == sigma(b)
    # The error for such a pair is k / (2*b)
    # To minimize error for a fixed k, we want to maximize b
    
    # So for each k, we should check b from the maximum valid value downward
    # until we find a match or run out of time
    
    found = False
    max_k_to_check = 1000  # safety limit on k
    
    for k in range(1, max_k_to_check + 1):
        if time.time() > deadline - 3:
            break
        
        # Case 1: 2a - 3b = k
        # a = (3b + k) // 2, requires (3b + k) % 2 == 0
        # b parity: 3b + k even => b + k even (since 3b has same parity as b)
        # So b % 2 == k % 2
        
        # b range: max(1, ceil((2-k)/3)) to min(MAX_N, floor((2*MAX_N - k)/3))
        b_min1 = max(1, math.ceil((2 - k) / 3))
        b_max1 = min(MAX_N, math.floor((2 * MAX_N - k) / 3))
        
        if b_min1 <= b_max1:
            # Check b from b_max1 downward (to get largest b for smallest error)
            # But we can't check all b values for large ranges.
            # Let's check a limited number of b values near the top.
            # Actually, for small k, the range is huge (up to ~10^7).
            # We need a smarter approach.
            
            # Alternative: iterate a instead?
            # a = (3b+k)/2 => b = (2a-k)/3
            # 2a-k must be divisible by 3 and positive
            # a range: from ceil((k+3)/2) to MAX_N
            # b = (2a-k)//3 must be in [1, MAX_N]
            
            # Let's just check a sample of b values near the maximum.
            # Or better: iterate a from MAX_N downward and check if b is valid.
            
            # For 2a - 3b = k: b = (2a - k) / 3
            # 2a - k must be >= 3 and divisible by 3
            # a must satisfy: 2a - k >= 3 => a >= (k+3)/2
            # 2a - k <= 3*MAX_N => a <= (3*MAX_N + k)/2
            # Also a <= MAX_N
            # And b >= 1 => 2a - k >= 3 => a >= ceil((k+3)/2)
            
            a_min_c1 = max(1, math.ceil((k + 3) / 2))
            a_max_c1 = min(MAX_N, math.floor((3 * MAX_N + k) / 2))
            
            if a_min_c1 <= a_max_c1:
                # Check a from a_max_c1 downward
                # 2a - k must be divisible by 3
                # 2a ≡ k (mod 3) => a ≡ 2k (mod 3) since 2*2=4≡1 mod 3... 
                # Actually 2a ≡ k (mod 3). Multiply by 2 (inverse of 2 mod 3): a ≡ 2k (mod 3)
                # 2*2 = 4 ≡ 1, so inverse of 2 is 2.
                # So a ≡ 2k (mod 3)
                
                a_start = a_max_c1
                # Adjust a_start to satisfy a ≡ 2k (mod 3)
                rem = a_start % 3
                target_rem = (2 * k) % 3
                diff = (target_rem - rem) % 3
                a_start = a_start - diff
                
                # Check up to some limit of a values
                count_checked = 0
                max_checks = 50000  # limit per k to avoid timeout
                
                for a in range(a_start, a_min_c1 - 1, -3):
                    if time.time() > deadline - 3:
                        break
                    count_checked += 1
                    if count_checked > max_checks:
                        break
                    
                    b = (2 * a - k) // 3
                    if b < 1 or b > MAX_N:
                        continue
                    if a == b:
                        continue
                    if 2 * a == 3 * b:
                        continue
                    
                    if sigma[a] == sigma[b]:
                        # Found a valid pair
                        # error = k / (2*b)
                        # Compare with best: k/(2*b) < best_err_num/(2*best_b)
                        # => k * best_b < best_err_num * b
                        if k * best_b < best_err_num * b:
                            best_err_num = k
                            best_a = a
                            best_b = b
                            found = True
                        break  # For this k, we found the best (largest b) for this direction
                        # Actually we should continue to check if there's a larger b?
                        # No, we're going downward in a, so b = (2a-k)/3 is also decreasing.
                        # Wait, as a decreases, b decreases. So the first valid one has the largest b.
                        # So we can break.
        
        # Case 2: 3b - 2a = k (i.e., 2a - 3b = -k)
        # b = (2a + k) / 3, requires (2a + k) % 3 == 0
        # a range: 1 <= a <= MAX_N
        # b = (2a+k)/3 must be in [1, MAX_N]
        # 2a + k >= 3 => a >= (3-k)/2
        # 2a + k <= 3*MAX_N => a <= (3*MAX_N - k)/2
        
        a_min_c2 = max(1, math.ceil((3 - k) / 2))
        a_max_c2 = min(MAX_N, math.floor((3 * MAX_N - k) / 2))
        
        if a_min_c2 <= a_max_c2:
            # 2a + k ≡ 0 (mod 3) => 2a ≡ -k (mod 3) => a ≡ -2k (mod 3)
            # -2 ≡ 1 (mod 3), so a ≡ k (mod 3)
            
            a_start = a_max_c2
            rem = a_start % 3
            target_rem = k % 3
            diff = (target_rem - rem) % 3
            a_start = a_start - diff
            
            count_checked = 0
            max_checks = 50000
            
            for a in range(a_start, a_min_c2 - 1, -3):
                if time.time() > deadline - 3:
                    break
                count_checked += 1
                if count_checked > max_checks:
                    break
                
                b = (2 * a + k) // 3
                if b < 1 or b > MAX_N:
                    continue
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue
                
                if sigma[a] == sigma[b]:
                    if k * best_b < best_err_num * b:
                        best_err_num = k
                        best_a = a
                        best_b = b
                        found = True
                    break
    
    # If we found a better pair, return it. Otherwise return the default.
    # The default [11, 6] has sigma(11)=12, sigma(6)=12. |11/6 - 3/2| = |22-18|/12 = 4/12 = 1/3
    # Wait, |a/b - 3/2| = |2a - 3b| / (2b). For a=11, b=6: |22-18|/(12) = 4/12 = 1/3.
    
    # Verify our best pair
    if best_a != 11 or best_b != 6:
        if sigma[best_a] != sigma[best_b]:
            return [11, 6]
        if 2 * best_a == 3 * best_b:
            return [11, 6]
    
    return [int(best_a), int(best_b)]
