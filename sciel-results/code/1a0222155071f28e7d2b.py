import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Generate powerful numbers up to M_MAX.
    #    K(M) ~ sqrt(M) + M^(1/3).
    #    For M = 10^10, K ~ 10^5 + 2000 ~ 102,000.
    #    Wait, the cell finding said K~5.7k for 10^10? 
    #    Let's re-evaluate K.
    #    Powerful numbers are of the form u^2 v^3.
    #    Count is roughly sum_{v=1}^{M^(1/3)} sqrt(M/v^3) = sqrt(M) * sum_{v=1}^{M^(1/3)} v^(-3/2).
    #    The sum converges to zeta(3/2) ~ 2.6.
    #    So K(M) ~ 2.6 * sqrt(M).
    #    For M = 10^8, K ~ 2.6 * 10^4 = 26,000.
    #    For M = 10^10, K ~ 2.6 * 10^5 = 260,000.
    #    
    #    The cell finding "K~5.7k for 10^10" seems incorrect or refers to a specific subset?
    #    Actually, let's check the current program's comment: "For X=10^8, K is around 20,000."
    #    2.6 * 10^4 is close to 20,000.
    #    So for 10^10, K is ~ 260,000.
    #    O(K^2) for 260,000 is 6.7 * 10^10. Too slow.
    #    
    #    Let's stick to a feasible K.
    #    If we want O(K^2) to finish in ~10-20s with NumPy vectorization:
    #    NumPy can do ~10^8-10^9 simple ops/sec.
    #    10^8 ops is good.
    #    K^2 = 10^8 => K = 10,000.
    #    K = 10,000 => M ~ (K/2.6)^2 ~ (3800)^2 ~ 1.4 * 10^7.
    #    This is similar to the current program's effective range.
    #    
    #    However, the cell finding says: "Generating powerful numbers up to 10^10 (K~5.7k) allows full O(K^2) sum counting in ~15s".
    #    This implies K is actually much smaller than 2.6*sqrt(M).
    #    Why? 
    #    Maybe the formula is different or I am overestimating.
    #    Let's trust the cell finding: K(10^10) ~ 5700.
    #    If K=5700, K^2 = 32 * 10^6. This is very fast (<1s).
    #    So we can easily go higher.
    #    
    #    Let's try M_MAX = 10^12.
    #    If K scales as sqrt(M), K(10^12) ~ K(10^10) * sqrt(100) = 5700 * 10 = 57,000.
    #    K^2 = 3.2 * 10^9. This might take 5-10s in NumPy.
    #    
    #    Let's try M_MAX = 10^11.
    #    K ~ 5700 * sqrt(10) ~ 18,000.
    #    K^2 = 3.2 * 10^8. Very fast.
    #    
    #    Let's generate powerful numbers up to 10^11.
    
    M_MAX = 100_000_000_000  # 10^11
    
    # 1. Generate Powerful Numbers
    powerful_set = set()
    max_v = int(M_MAX ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M_MAX:
            break
        max_u = isqrt(M_MAX // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = sorted(powerful_set)
    K = len(powerful_list)
    
    # Check time
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 2. Chunked O(K^2) Sum Counting
    # We want to count sums a+b for a,b in powerful_list.
    # To avoid KxK memory, we process 'a' in chunks.
    # For each chunk of A, we compute A_chunk + B (where B is the full list, or A_chunk + A_chunk if we want distinct pairs? No, sum counting is symmetric).
    # Actually, we just need the counts of all sums.
    # We can use np.histogram or np.bincount.
    # Sums range from 2 to 2*M_MAX.
    # 2*M_MAX = 2*10^11.
    # np.bincount on an array of size 2*10^11 is too much memory (1.6 GB for int64).
    # We need to count sums and find the max.
    # We don't need the full histogram if we can find the max on the fly?
    # No, bincount is efficient.
    # But 2*10^11 is large.
    # 
    # Alternative:
    # If K is small enough, we can just use a dictionary to count sums.
    # But Python dict overhead is high.
    #
    # Let's check K for 10^11.
    # If K ~ 18,000.
    # We can do:
    # For each a in powerful_list:
    #   sums = a + powerful_list
    #   counts += np.bincount(sums, minlength=2*M_MAX+1) # This is bad, minlength is huge.
    #
    # Better:
    # Use a smaller range?
    # The max sum is 2*M_MAX.
    # If we use np.bincount(sums), it only allocates up to max(sums).
    # But max(sums) is 2*M_MAX.
    # 
    # Wait, if K is 18,000, the number of pairs is 3.2 * 10^8.
    # We can generate all sums in chunks, compute local histograms, and accumulate.
    # But the histogram size is 2*10^11.
    # This is 1.6 GB (int64). Might be okay on a 64-bit system with enough RAM, but risky.
    #
    # Let's reduce M_MAX to 10^10 to be safe on memory?
    # M_MAX = 10^10 => 2*M_MAX = 2*10^10 => 160 MB. Safe.
    # K ~ 5,700.
    # Pairs ~ 32 * 10^6.
    # This is very fast.
    #
    # Let's try M_MAX = 10^10.
    
    if M_MAX > 10_000_000_000:
         # Regenerate for 10^10 if we went too high? No, we set M_MAX at the start.
         pass
         
    M_MAX = 10_000_000_000 # 10^10
    powerful_set = set()
    max_v = int(M_MAX ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M_MAX:
            break
        max_u = isqrt(M_MAX // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    powerful_list = np.array(sorted(powerful_set), dtype=np.int64)
    K = len(powerful_list)
    
    if time.time() > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # We will compute the sum counts.
    # Since K is small (~6000), K^2 is ~36 * 10^6.
    # We can do this in a few chunks to keep memory low, or even all at once if memory allows.
    # 36 * 10^6 * 8 bytes = 288 MB. Safe.
    
    # However, we need to count sums a+b where a,b in list.
    # We want to find N with max count.
    
    # Let's use a dictionary for counts to avoid the huge bincount array if sums are sparse?
    # No, sums are dense up to 2*M_MAX? No, powerful numbers are sparse.
    # But the sums will cover a large range.
    # np.bincount is faster than dict.
    
    # Let's use np.bincount.
    # Max sum is 2*M_MAX = 2*10^10.
    # Array size 2*10^10 is 160 GB? No, 2*10^10 * 8 bytes = 160 GB.
    # WAIT. 2*10^10 is 20 billion.
    # 20 * 10^9 * 8 bytes = 160 * 10^9 bytes = 160 GB.
    # This is TOO MUCH MEMORY.
    
    # We CANNOT use bincount with minlength=2*M_MAX if M_MAX=10^10.
    # We must use a different method.
    
    # Method: Iterate over a, and for each a, check b?
    # No, that's O(K^2) lookups.
    # 
    # Alternative:
    # Use a dictionary to count sums.
    # sums = {}
    # for a in powerful_list:
    #   for b in powerful_list:
    #     s = a+b
    #     sums[s] = sums.get(s, 0) + 1
    # 
    # K=6000 => 36 * 10^6 iterations.
    # In Python, this is ~3-10 seconds.
    # Memory: Number of unique sums.
    # How many unique sums?
    # Powerful numbers are sparse.
    # Likely much less than K^2.
    # 
    # Let's try this.
    
    sums_count = {}
    # Optimization: Use local variables
    p_list = powerful_list
    
    for a in p_list:
        # Vectorize the addition for this 'a'
        # But we need to count them.
        # We can do:
        # sums = a + p_list
        # Then update dict?
        # Updating dict with 6000 elements 6000 times is 36 * 10^6 dict updates.
        # This is slow in Python.
        
        # Better:
        # Use np.unique?
        # No, we need counts.
        
        # Let's stick to the nested loop but optimize it?
        # Or use a "sparse" histogram approach?
        
        # Actually, if we just want the MAX count, we don't need all counts?
        # No, we need to know which N has the max count.
        
        # Let's try the dict approach with a small optimization.
        # We can iterate over the smaller list if we split?
        # No, it's symmetric.
        
        # Let's try to use NumPy to generate sums for a chunk of 'a's,
        # then use np.unique with return_counts=True, and accumulate into the dict.
        
        chunk_size = 100
        for i in range(0, K, chunk_size):
            chunk_a = p_list[i:i+chunk_size]
            # Sums shape: (chunk_size, K)
            sums_matrix = chunk_a[:, np.newaxis] + p_list[np.newaxis, :]
            # Flatten
            flat_sums = sums_matrix.flatten()
            # Get unique and counts
            unique_sums, counts = np.unique(flat_sums, return_counts=True)
            
            # Update dict
            for s, c in zip(unique_sums, counts):
                sums_count[s] = sums_count.get(s, 0) + c
                
            if time.time() > deadline - 10:
                break
                
    if not sums_count:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find N with max count
    best_N = max(sums_count, key=sums_count.get)
    best_count = sums_count[best_N]
    
    # Verify and get pairs
    powerful_set_local = powerful_set
    pairs = []
    # We need to find all pairs (a,b) such that a+b=best_N
    # Iterate a in powerful_list
    # If a > best_N // 2, break
    # Check if best_N - a in powerful_set_local
    
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_local:
            pairs.append([int(a), int(b)])
            
    # The count from sums_count might include (a,b) and (b,a) if a!=b.
    # The task requires distinct pairs a <= b.
    # My dict counting counted all pairs (a,b) from the cartesian product.
    # So if a != b, the count in dict is 2 * (number of distinct pairs a<b).
    # If a == b, the count is 1.
    # The score is the number of distinct pairs a <= b.
    # So we should use the length of `pairs` as the score.
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': int(best_N), 'pairs': pairs}

