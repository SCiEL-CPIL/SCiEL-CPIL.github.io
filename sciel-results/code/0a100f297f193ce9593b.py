import time
from math import gcd
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline
    
    n = 30
    # Calculate LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
        
    # Weights
    W = [L // (i+1) for i in range(n)]
    
    mid = n // 2
    W1 = W[:mid]
    W2 = W[mid:]
    
    # Generate all sums for first half (coeffs -1, 1)
    # There are 2^15 = 32768 possibilities
    # We store (sum_value, coeffs_tuple)
    # To save memory, we can store sum_value and bitmask
    
    def generate_sums(weights):
        m = len(weights)
        sums = []
        # Iterative generation
        # Start with [(0, 0)] where 0 is bitmask
        # For each weight, we branch: -w (bit 0) and +w (bit 1)
        # Actually, let's just use product([-1, 1], repeat=m)
        # 32768 * 15 operations is ~ 500k. Very fast.
        for coeffs in product([-1, 1], repeat=m):
            s = sum(c * w for c, w in zip(coeffs, weights))
            sums.append((s, coeffs))
        return sums

    sums1 = generate_sums(W1)
    sums2 = generate_sums(W2)
    
    if time.time() > deadline - 10:
        # Fallback: return all 1s? Sum is not 0.
        # Denom will be large, but valid.
        return [1] * n

    # Get divisors of L
    def get_divisors(n):
        divisors = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
        return sorted(list(divisors), reverse=True)
        
    divisors = get_divisors(L)
    
    # For each divisor d, check if there exists n1, n2 such that (n1+n2) % d == 0 and n1+n2 != 0
    # To speed up, we can pre-compute residues for each d?
    # No, d changes.
    # But we can just iterate.
    
    best_coeffs = None
    best_d = 0
    
    # Optimization: 
    # We want to maximize G = gcd(S, L).
    # We iterate d in descending order.
    # The first d for which we find a valid non-zero sum is the max G.
    
    for d in divisors:
        # Check if there is a pair with sum divisible by d
        # Residues of sums1 modulo d
        # Residues of sums2 modulo d
        
        # We need to find r1, r2 such that (r1 + r2) % d == 0
        # i.e., r2 == -r1 % d
        
        # Build set of (residue, index) for sums1
        # If multiple indices have same residue, we might need to try multiple
        # But usually one is enough unless sum is 0
        
        rem1_map = {}
        for i, (s, c) in enumerate(sums1):
            r = s % d
            if r not in rem1_map:
                rem1_map[r] = []
            rem1_map[r].append(i)
            
        rem2_map = {}
        for i, (s, c) in enumerate(sums2):
            r = s % d
            if r not in rem2_map:
                rem2_map[r] = []
            rem2_map[r].append(i)
        
        found = False
        # Iterate over residues in rem1_map
        for r1, idxs1 in rem1_map.items():
            target_r2 = (-r1) % d
            if target_r2 in rem2_map:
                idxs2 = rem2_map[target_r2]
                # Try combinations
                # We just need ONE non-zero sum
                for i1 in idxs1:
                    s1_val = sums1[i1][0]
                    c1 = sums1[i1][1]
                    for i2 in idxs2:
                        s2_val = sums2[i2][0]
                        c2 = sums2[i2][1]
                        total_sum = s1_val + s2_val
                        if total_sum != 0:
                            # Found valid pair
                            coeffs = list(c1) + list(c2)
                            best_coeffs = coeffs
                            best_d = d
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        
        if found:
            break
            
    if best_coeffs is None:
        # Should not happen, but fallback
        return [1] * n
        
    return best_coeffs
