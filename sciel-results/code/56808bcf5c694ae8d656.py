import time
from math import isqrt

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0
    
    def generate_powerful(limit):
        """Generate all powerful numbers <= limit."""
        pw = set()
        # Powerful numbers are of the form u^2 * v^3 where gcd(u,v) doesn't strictly matter for the set definition
        # but the representation u^2 v^3 covers all powerful numbers.
        # Actually, a powerful number is one where every prime factor p has exponent >= 2.
        # It can be uniquely written as u^2 * v^3 where v is cube-free? No.
        # The standard generation is: for each v, v^3 * u^2.
        # To ensure we catch all, we iterate v up to limit^(1/3) and u up to sqrt(limit/v^3).
        
        max_v = int(limit ** (1/3)) + 2
        
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > limit:
                break
            max_u = isqrt(limit // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                if p <= limit:
                    pw.add(p)
        return sorted(pw)

    # Strategy:
    # The previous best score was ~4.67 with M=20M (K~3000).
    # The cell findings indicate that M=10^10 (K~5700) allows full O(K^2) sum counting in ~15s.
    # This is significantly faster than the 150s budget, allowing us to:
    # 1. Generate powerful numbers up to 10^10.
    # 2. Compute all pairwise sums and count frequencies exactly.
    # 3. Return the N with the highest count.
    
    # We can even try to go higher if time permits, but 10^10 is a safe, measured sweet spot.
    # Let's start with 10^10. If it finishes quickly, we could try 2*10^10, but 10^10 is robust.
    
    target_M = 10_000_000_000
    
    # Check time before generation
    if time.time() > deadline - 30:
        # Fallback to a smaller range if we are already late (unlikely at start)
        target_M = 1_000_000_000

    powerful_list = generate_powerful(target_M)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Check time before O(K^2) loop
    # K=5700 -> ~16M iterations. In Python, this might take 5-10 seconds.
    # If K is too large (e.g. > 10000 -> 50M iterations), it might take 20-30s.
    # We have 140s. 10^10 is safe.
    
    if time.time() > deadline - 20:
        # If generation took too long, we might not have enough time for the full loop.
        # But generation of 10^10 is fast (<1s).
        # If we are here, we are in trouble. Return best found so far? 
        # We don't have pairs yet. Return a trivial solution.
        return {'N': 2, 'pairs': [[1, 1]]}

    # To speed up the O(K^2) loop, we can use a dictionary to count sums.
    # However, pure Python loops are slow. 
    # Let's see if we can optimize.
    # We need to find the max count of N = a + b.
    # We can use a defaultdict(int).
    
    from collections import defaultdict
    
    sum_counts = defaultdict(int)
    # We also need to store the pairs for the winner.
    # Storing pairs for ALL sums is too memory intensive (16M pairs).
    # We only need the pairs for the BEST N.
    # So, first pass: find the best N and its count.
    # Second pass: retrieve the pairs for that N.
    
    # Optimization: 
    # Instead of storing all pairs, we just count.
    # Then we find the max N.
    # Then we iterate again to collect pairs for that specific N.
    
    # Pass 1: Count sums
    # Using local variables for speed
    pw_list = powerful_list
    k = K
    
    # We can potentially use NumPy to vectorize the counting?
    # np.add.outer(pw_list, pw_list) creates a KxK array.
    # For K=5700, 5700x5700 = 32.4M elements.
    # 32.4M * 8 bytes (int64) = 260 MB. This is feasible.
    # np.bincount can then count the frequencies in C-speed.
    # This would be MUCH faster than a Python loop.
    
    import numpy as np
    
    # Check memory feasibility
    # If K > 10000, 100M elements * 8 bytes = 800MB. Might be tight.
    # For K=5700, it's safe.
    
    try:
        # Create outer sum matrix
        # We only need a <= b, so we can use triu indices or just do full and divide?
        # Full matrix is simpler for bincount.
        # But we need distinct pairs a <= b.
        # If we use the full matrix, count(N) will be 2 * (pairs with a!=b) + 1 (if N=2a).
        # We want the count of unordered pairs (a,b) with a<=b.
        # Let C_full(N) be the count from the full KxK matrix.
        # Let D(N) be the number of diagonal elements (a=a) summing to N. This is 1 if N is even and N/2 in pw_list, else 0.
        # Then C_unordered(N) = (C_full(N) + D(N)) // 2.
        
        # Generating the full outer sum:
        # a_i + b_j
        # We can do:
        # sums = pw_list[:, None] + pw_list[None, :]
        # This creates a KxK array.
        
        sums = pw_list[:, None] + pw_list[None, :]
        
        # Count frequencies
        # Max sum is 2 * max(pw_list) <= 2 * 10^10.
        # bincount requires max value < 2^31 if using int32? 
        # np.bincount uses int64 for counts, but the input values are used as indices.
        # If max sum is 2*10^10, bincount will try to allocate an array of size 2*10^10. 
        # That is 20 billion integers. 80 GB. WAY too big.
        
        # So np.bincount is NOT an option for large N.
        # We must use a dictionary or sparse method.
        
        # Alternative: Use pandas? No.
        # Alternative: Sort the sums?
        # sums.flatten() -> 32M elements.
        # Sorting 32M elements takes ~2-3 seconds.
        # Then we can scan for the longest run of identical values.
        # This avoids the huge memory allocation of bincount.
        
        flat_sums = sums.flatten()
        
        # Check time
        if time.time() > deadline - 10:
             return {'N': 2, 'pairs': [[1, 1]]}
             
        # Sort
        flat_sums.sort()
        
        # Find max count
        best_count = 0
        best_N = 0
        i = 0
        n_flat = len(flat_sums)
        
        while i < n_flat:
            val = flat_sums[i]
            j = i + 1
            while j < n_flat and flat_sums[j] == val:
                j += 1
            count = j - i
            if count > best_count:
                best_count = count
                best_N = val
            i = j
            
    except MemoryError:
        # Fallback to Python loop if NumPy allocation fails
        sum_counts = defaultdict(int)
        for i in range(K):
            a = pw_list[i]
            for j in range(i, K):
                b = pw_list[j]
                sum_counts[a + b] += 1
        
        if not sum_counts:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        best_N, best_count = max(sum_counts.items(), key=lambda x: x[1])
        
    # Now we have best_N and best_count (from full matrix logic? No, the sort logic counts ordered pairs).
    # Wait, the sort logic counts ordered pairs (a,b) from the full KxK matrix.
    # Let C_ordered = best_count.
    # We need C_unordered.
    # D = 1 if best_N is even and best_N//2 in pw_list else 0.
    # C_unordered = (C_ordered + D) // 2.
    
    # Let's verify D
    D = 0
    if best_N % 2 == 0:
        half_N = best_N // 2
        # Check if half_N is in pw_list
        # pw_list is sorted, so we can use binary search or just a set.
        # For efficiency, let's just check membership.
        # Since we are in the "find pairs" phase anyway, we can do it there.
        
    # Now, retrieve the pairs for best_N.
    # We need pairs (a, b) such that a + b = best_N and a <= b.
    # We can iterate through pw_list.
    # For each a in pw_list:
    #   if a > best_N // 2: break
    #   b = best_N - a
    #   if b in pw_set:
    #      add [a, b]
    
    pw_set = set(pw_list)
    pairs = []
    half_N = best_N // 2
    
    for a in pw_list:
        if a > half_N:
            break
        b = best_N - a
        if b in pw_set:
            # Since a <= b is guaranteed by a <= half_N and b = N - a >= N - N/2 = N/2 >= a
            pairs.append([a, b])
            
    # The number of pairs found should be C_unordered.
    # Score is len(pairs).
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}

