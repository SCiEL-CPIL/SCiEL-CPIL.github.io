import time
import numpy as np
from math import isqrt
from collections import Counter

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers <= limit.
    A number n is powerful if n = x^2 * y^3 where y is square-free.
    Equivalently, all prime factors have exponent >= 2.
    """
    if limit < 1:
        return []
    
    powerfuls = set()
    
    # Max value for y (cube part)
    max_y = int(limit ** (1/3)) + 2
    # Max value for x (square part)
    # x^2 * 1^3 <= limit => x <= sqrt(limit)
    
    # We iterate y from 1 to max_y
    # For each y, we check if y is square-free?
    # Actually, the form x^2 * y^3 generates all powerful numbers.
    # But we must ensure y is square-free to get unique representations?
    # No, we just need the set of numbers. Duplicates in set are fine.
    # However, if y is not square-free, say y=4, y^3=64=8^2.
    # Then x^2 * 64 = (4x)^2. This is a square, which is powerful.
    # So we can just iterate all y.
    
    for y in range(1, max_y + 1):
        y3 = y * y * y
        if y3 > limit:
            break
        # x^2 <= limit / y3
        max_x = isqrt(limit // y3)
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerfuls.add(n)
                
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Generate powerful numbers up to 10^9
    # This gives approx 31,000 numbers
    # Sums will be up to 2*10^9
    L = 10**9
    pow_list = generate_powerful_numbers(L)
    
    if not pow_list:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    P = np.array(pow_list, dtype=np.int64)
    n = len(P)
    
    # If n is small, we can do it all at once?
    # 31,000^2 is ~10^9, too big for memory (8GB).
    # Use blocking.
    
    counts = Counter()
    block_size = 2000
    
    for i in range(0, n, block_size):
        if time.time() > deadline - 10:
            break
            
        end = min(i + block_size, n)
        # We want pairs (a, b) with a <= b.
        # Let block be P[i:end].
        # We sum with P[j] where j >= i? 
        # Actually, to avoid duplicates and ensure a<=b, 
        # we can just sum block with ALL P, then filter a<=b.
        # But if we sum block P[i:end] with P[0:n], we might count (a,b) where a in block, b in P.
        # If a > b, we discard.
        # But we might miss pairs where a is in a later block and b is in an earlier block?
        # No, if we iterate i from 0 to n, and for each i, we consider a=P[i], 
        # and b from P[i] to P[n-1] (since a<=b implies index_a <= index_b if sorted? 
        # Not necessarily index, but value. Since sorted, P[i] <= P[j] iff i <= j).
        
        # So, for block P[i:end], we should sum with P[i:n] to ensure a <= b and avoid double counting.
        # Wait, if a is in block i, and b is in block k > i, that's fine.
        # If a is in block i, and b is in block i (same block), we need b >= a.
        
        # Let's do:
        # For each block start i:
        #   A = P[i:end]
        #   B = P[i:n]  # All valid b's for any a in A such that a <= b is possible?
        #   Actually, if a = P[i], b must be >= P[i].
        #   If a = P[end-1], b must be >= P[end-1].
        #   So for the whole block, B = P[i:n] is a superset of valid b's.
        #   We compute S = A[:, None] + B[None, :]
        #   Mask: A[:, None] <= B[None, :]
        
        A = P[i:end]
        B = P[i:n]
        
        # Vectorized addition
        # Shape: (len(A), len(B))
        sums = A[:, None] + B[None, :]
        
        # Mask
        mask = A[:, None] <= B[None, :]
        valid_sums = sums[mask]
        
        if len(valid_sums) > 0:
            # Use np.unique to count
            unique_vals, unique_counts = np.unique(valid_sums, return_counts=True)
            # Update global counter
            for val, cnt in zip(unique_vals, unique_counts):
                counts[int(val)] += int(cnt)
                
    # Find best N
    if not counts:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N, best_count = max(counts.items(), key=lambda x: x[1])
    
    # Reconstruct pairs for best_N
    pairs = []
    # pow_list is sorted
    # a + b = best_N => b = best_N - a
    # Iterate a in pow_list
    # If a > best_N // 2, break
    # Check if b is in pow_list (using binary search or set)
    
    pow_set = set(pow_list)
    for a in pow_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in pow_set:
            # Since pow_list is sorted and we iterate a, and a <= best_N//2, 
            # and b = N-a >= a, so a <= b is satisfied.
            # Also b <= L? 
            # If b > L, it wasn't in pow_list, so b not in pow_set.
            # So this only finds pairs where both a and b are <= L.
            # This is consistent with our counting.
            pairs.append([a, b])
            
    if not pairs:
        # Should not happen if counts > 0, but just in case
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
