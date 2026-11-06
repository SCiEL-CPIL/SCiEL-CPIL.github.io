import numpy as np
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using square-free b enumeration."""
    # A powerful number has the form a^2 * b^3 where b is square-free
    max_b = int(limit ** (1.0 / 3.0)) + 2
    
    # Sieve for square-free numbers up to max_b
    is_sq_free = [True] * (max_b + 1)
    if max_b >= 0:
        is_sq_free[0] = False
    if max_b >= 1:
        is_sq_free[1] = True
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        for j in range(sq, max_b + 1, sq):
            is_sq_free[j] = False
    
    powerful_list = []
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b * b * b
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful_list.append(val)
    
    powerful_arr = np.array(powerful_list, dtype=np.int64)
    return powerful_arr

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s
    
    # Generate powerful numbers up to a limit
    # The key insight: we want to find N with many pairs (a, b) where a+b=N and both are powerful.
    # The number of powerful numbers up to x is about x^(1/3) * C for some constant.
    # For N up to 10^18, we need powerful numbers up to 10^18.
    
    # Strategy: generate powerful numbers up to some limit, then for each powerful number a,
    # compute N = a + b for all powerful b, and count occurrences.
    # But that's O(|P|^2) which is too much.
    
    # Better: For each powerful number a, and for each powerful number b >= a,
    # compute s = a + b. Count how many times each s appears. The s with the highest count
    # is our best N.
    
    # Let's try different limits for powerful numbers and see what we can achieve.
    # With limit = 10^8, we get about 10^8^(1/3) = ~464 powerful numbers (roughly).
    # Actually the count is more like sum over square-free b of sqrt(limit/b^3).
    
    # Let's start with a moderate limit and increase if time allows.
    
    best_result = None
    best_score = 0
    
    limits_to_try = [10**6, 10**7, 10**8, 5*10**7, 2*10**8, 5*10**8, 10**9]
    
    for limit in limits_to_try:
        if time.time() - start_time > deadline - 10:
            break
            
        powerful = _generate_powerful(limit)
        n_pow = len(powerful)
        
        if n_pow == 0:
            continue
        
        # For counting representations, we need pairs (a, b) with a <= b, a + b = N.
        # We can use the approach: for each a in powerful, consider all b in powerful with b >= a,
        # compute s = a + b, and count.
        
        # Use vectorized approach per row to keep memory manageable.
        # But n_pow can be large. Let's think about the count.
        # For limit = 10^8, n_pow is roughly... let's estimate.
        # The number of powerful numbers <= x is ~ 1.3 * x^(1/3) + O(x^(1/4))
        # So for 10^8, ~1.3 * 464 ~ 600. That's manageable.
        # For 10^9, ~1.3 * 1000 = 1300. Still manageable.
        # For 10^12, ~1.3 * 10000 = 13000. Pairwise would be 13000^2 = 169 million, too much.
        
        if n_pow > 20000:
            # Too many for pairwise
            continue
        
        # Compute all pairwise sums a + b where a <= b
        # To save memory and time, process in chunks or use a clever approach.
        
        # For n_pow up to ~20000, n_pow^2/2 = 200 million pairs. That's too many for memory.
        # Let's cap n_pow at something reasonable for pairwise.
        
        if n_pow > 10000:
            # Use a different approach: for each a, find b in powerful such that b >= a and b <= limit.
            # We want to count, for each possible sum s, how many pairs (a, b) give that sum.
            # This is a convolution-like problem.
            
            # Alternative: for each powerful number a, and each powerful number b with b >= a,
            # we add to count[a+b]. But we can't store an array of size limit.
            
            # Use a dictionary.
            from collections import defaultdict
            counts = defaultdict(int)
            pow_set = set(powerful.tolist())
            
            for i in range(n_pow):
                a = powerful[i]
                # b must be >= a and in powerful
                for j in range(i, n_pow):
                    b = powerful[j]
                    if b < a:
                        continue
                    s = a + b
                    if s <= 10**18:
                        counts[s] += 1
                if time.time() - start_time > deadline - 5:
                    break
            
            if not counts:
                continue
            
            best_s = max(counts, key=counts.get)
            best_count = counts[best_s]
            
            if best_count > best_score:
                # Find all pairs that sum to best_s
                pairs = []
                for a in powerful:
                    b = best_s - a
                    if a <= b and b in pow_set:
                        pairs.append([int(a), int(b)])
                pairs.sort()
                # Deduplicate
                seen = set()
                unique_pairs = []
                for p in pairs:
                    key = (p[0], p[1])
                    if key not in seen:
                        seen.add(key)
                        unique_pairs.append(p)
                
                best_score = best_count
                best_result = {'N': int(best_s), 'pairs': unique_pairs}
        else:
            # n_pow <= 10000, can do pairwise more efficiently
            from collections import defaultdict
            counts = defaultdict(int)
            pow_set = set(powerful.tolist())
            
            for i in range(n_pow):
                a = powerful[i]
                for j in range(i, n_pow):
                    b = powerful[j]
                    s = a + b
                    if s <= 10**18:
                        counts[s] += 1
                if time.time() - start_time > deadline - 5:
                    break
            
            if not counts:
                continue
            
            best_s = max(counts, key=counts.get)
            best_count = counts[best_s]
            
            if best_count > best_score:
                pairs = []
                for a in powerful:
                    b = best_s - a
                    if a <= b and b in pow_set:
                        pairs.append([int(a), int(b)])
                pairs.sort()
                seen = set()
                unique_pairs = []
                for p in pairs:
                    key = (p[0], p[1])
                    if key not in seen:
                        seen.add(key)
                        unique_pairs.append(p)
                
                best_score = best_count
                best_result = {'N': int(best_s), 'pairs': unique_pairs}
    
    if best_result is None:
        # Fallback: small case
        return {'N': 3, 'pairs': [[1, 2]]}
    
    return best_result
