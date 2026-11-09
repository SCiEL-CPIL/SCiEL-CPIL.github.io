<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import bisect
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 120.0
    
    def time_exceeded():
        return time.time() >= DEADLINE

    def get_primes(limit):
        if limit < 2: return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Generate powerful numbers up to L
    L = 10**7
    primes = get_primes(int(L**0.5) + 1)
    
    powerful = [1]
    powerful_set = set([1])
    
    # Iterative generation using a queue-like approach
    # Start with 1, and for each powerful number, multiply by p^2 for primes p
    # where p is the next prime after the largest prime factor used
    # This is tricky iteratively. Let's use a different approach:
    # Generate all numbers of the form prod p_i^e_i with e_i >= 2
    # by iterating over primes and building up.
    
    # Use a list to store powerful numbers and build incrementally
    # For each prime p, for each existing powerful number x, add x * p^2, x * p^3, ...
    # But we need to ensure we don't miss combinations.
    
    # Better: use the standard method - for each powerful number, try multiplying by p^2
    # for each prime p >= last_prime_factor. But tracking last_prime_factor is complex.
    
    # Alternative: generate all powerful numbers by checking each number? Too slow.
    
    # Let's use a recursive approach but with explicit stack to avoid recursion limit
    def generate_powerful_iterative():
        pn = [1]
        pn_set = {1}
        # For each powerful number, try to extend by multiplying by p^k
        # We need to ensure unique generation. Use the property that each powerful
        # number has a unique prime factorization with exponents >= 2.
        # Generate by: for each prime p, for each existing pn that doesn't have p as factor,
        # add pn * p^2, pn * p^3, ...
        # But "doesn't have p as factor" is hard to check quickly.
        
        # Simpler: use the DFS but with a stack
        stack = [(0, 1)]
        while stack:
            idx, val = stack.pop()
            if val > L:
                continue
            if val > 1:
                pn_set.add(val)
            
            for i in range(idx, len(primes)):
                p = primes[i]
                nv = val * p * p
                if nv > L:
                    break
                stack.append((i + 1, nv))
                while nv * p <= L:
                    nv *= p
                    pn_set.add(nv)
                    stack.append((i + 1, nv))
                    if time_exceeded():
                        return pn_set
        
        return pn_set
    
    powerful_set = generate_powerful_iterative()
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pn_list = sorted(powerful_set)
    
    # Use sum-counting approach: for each pair (i,j) with i<=j, count[pn[i]+pn[j]] += 1
    # Then find the N with max count, and reconstruct pairs.
    # Limit to first K powerful numbers to keep O(K^2) manageable.
    
    # Estimate K: if L=10^7, P ~ 10^(7*2/3) = 10^(14/3) ~ 10^4.67 ~ 46000
    # K^2 = 46000^2 = 2.1e9 which is too slow.
    # So limit K to ~3000: K^2 = 9e6, fast.
    
    K = min(len(pn_list), 3000)
    small_pn = pn_list[:K]
    
    sum_count = defaultdict(int)
    sum_pairs = defaultdict(list)
    
    for i in range(K):
        a = small_pn[i]
        for j in range(i, K):
            b = small_pn[j]
            s = a + b
            if s <= 2 * L:  # Both a and b <= L, so s <= 2L
                sum_count[s] += 1
                sum_pairs[s].append([a, b])
    
    if not sum_count:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_count, key=sum_count.get)
    best_count = sum_count[best_N]
    best_pairs = sum_pairs[best_N]
    
    # Verify: all pairs should have a <= b and a+b = best_N, and a,b powerful
    # They are by construction.
    
    # Safety: ensure N >= 2
    if best_N < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin
    
    def time_exceeded():
        return time.time() >= DEADLINE

    def get_primes(limit):
        if limit < 2: return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Generate powerful numbers up to L
    # Powerful numbers: every prime factor appears with exponent >= 2
    # Count is ~ L^(2/3) * constant
    L = 2 * 10**8
    max_prime = int(L ** 0.5) + 2
    primes = get_primes(max_prime)
    
    powerful_set = set([1])
    
    # DFS to generate all powerful numbers
    # Each powerful number = product of p_i^(e_i) where e_i >= 2
    # We generate by choosing primes in increasing order
    def generate_powerful():
        stack = [(0, 1)]
        while stack:
            idx, val = stack.pop()
            for i in range(idx, len(primes)):
                p = primes[i]
                p2 = p * p
                if val * p2 > L:
                    break
                nv = val * p2
                powerful_set.add(nv)
                stack.append((i + 1, nv))
                # Continue with higher powers of p
                while nv * p <= L:
                    nv *= p
                    powerful_set.add(nv)
                    stack.append((i + 1, nv))
                if time_exceeded():
                    return
        # 1 is already in powerful_set
    
    generate_powerful()
    
    if time_exceeded():
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pn_list = sorted(powerful_set)
    n_pn = len(pn_list)
    
    # Use NumPy to compute pairwise sums efficiently
    # We want to find N that maximizes the number of pairs (a,b) with a<=b, a+b=N
    # where a,b are powerful numbers
    
    # Strategy: For each powerful number a, compute N - a for all candidate N values
    # But we don't know N in advance. Instead, compute all pairwise sums and count them.
    
    # For large n_pn, we can't store all O(n_pn^2) sums. 
    # Instead, use a smarter approach:
    # For each possible sum S, count how many pairs (a,b) with a<=b satisfy a+b=S.
    # This is equivalent to: for each a in pn_list, for each b in pn_list with b>=a, count[a+b]++.
    
    # With NumPy, we can do this in chunks to avoid memory issues.
    # But first, let's see how many powerful numbers we have.
    
    if n_pn < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pn_arr = np.array(pn_list, dtype=np.int64)
    
    # We'll compute sums in a memory-efficient way.
    # For each i, compute pn[i] + pn[j] for j >= i.
    # Sum all these into a histogram.
    
    # Max sum is 2*L
    max_sum = 2 * L
    # Use a dictionary for sparse counting if max_sum is too large for an array
    # 2*10^8 is manageable as an array (200M entries, ~800MB for int64 - too much)
    # Use a dictionary instead, or use a smaller approach.
    
    # Better approach: Since we're looking for the N with the most representations,
    # and powerful numbers are sparse, let's use a different strategy.
    
    # For each powerful number a, the complementary powerful numbers b = N - a must also be powerful.
    # So for a fixed N, the number of representations is the number of powerful a such that N-a is also powerful.
    
    # We can iterate over candidate N values and count representations.
    # But what N values to try? N must be a sum of two powerful numbers.
    
    # Alternative: Use the fact that if a and b are powerful, then a+b is even or odd depending on parity.
    # Most powerful numbers > 1 are even (since 2^2=4 divides them if they have factor 2) or odd.
    
    # Let me try a different structural approach:
    # 1. Generate powerful numbers up to L
    # 2. For each powerful number a, for each powerful number b >= a, record sum = a+b
    #    But do this in chunks using NumPy to speed up
    
    # Chunk-based pairwise sum counting
    chunk_size = 5000  # Number of i values to process at once
    sum_counts = defaultdict(int)
    
    for i_start in range(0, n_pn, chunk_size):
        i_end = min(i_start + chunk_size, n_pn)
        pn_chunk = pn_arr[i_start:i_end]
        
        for i_local in range(i_end - i_start):
            i = i_start + i_local
            a = pn_arr[i]
            # For j >= i, compute a + pn[j]
            pn_tail = pn_arr[i:]
            sums = a + pn_tail
            # Count each sum
            for s in sums:
                sum_counts[int(s)] += 1
        
        if time_exceeded():
            break
    
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the N with maximum count
    best_N = max(sum_counts, key=sum_counts.get)
    best_count = sum_counts[best_N]
    
    # Reconstruct pairs for best_N
    pn_set_check = set(pn_list)
    best_pairs = []
    for a in pn_list:
        b = best_N - a
        if a <= b and b in pn_set_check:
            best_pairs.append([a, b])
    
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the above approach still has an O(n_pn²) loop in Python for the inner `for s in sums` loop. Let me reconsider.

The key insight from the observations is that NumPy vectorization reduces the constant factor by ~100x. Let me use a proper NumPy approach where I compute all pairwise sums for a chunk of i values against all j >= i values, then use `np.bincount` or similar to count them efficiently.

