import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    # 1. Generate powerful numbers up to specific limits
    # Powerful number: every prime factor appears with exponent >= 2.
    # Generation method: p = u^2 * v^3 where u, v are integers.
    # This generates all powerful numbers uniquely? 
    # Actually, a powerful number can be written as u^2 v^3 in multiple ways if not careful?
    # No, the standard generation is:
    # For each v, v^3 <= M:
    #   For each u, u^2 <= M / v^3:
    #     Add u^2 * v^3
    # This generates all powerful numbers. Are they unique?
    # 36 = 6^2 * 1^3 = 1^2 * ? No.
    # 36 = 2^2 * 3^2.
    # u=6, v=1 -> 36*1 = 36.
    # u=2, v=? 2^2 * v^3 = 36 -> 4 v^3 = 36 -> v^3 = 9 (no).
    # It turns out this representation is not unique.
    # Example: 216 = 6^3 = 6^2 * 6^1? No.
    # 216 = 2^3 * 3^3.
    # u=6, v=2 -> 36 * 8 = 288.
    # u=1, v=6 -> 1 * 216 = 216.
    # u=6, v=1 -> 36.
    # Wait, 216 = 6^3. Is 6 powerful? No.
    # 216 = 2^3 * 3^3.
    # Can it be u^2 v^3?
    # If v=6, v^3 = 216. u=1 -> 216.
    # If v=2, v^3 = 8. u^2 = 27 (no).
    # If v=3, v^3 = 27. u^2 = 8 (no).
    # So uniqueness is not guaranteed by just iterating u,v.
    # We must use a set to remove duplicates.
    
    def generate_powerful(M):
        pw_set = set()
        # v^3 <= M
        max_v = int(M ** (1/3)) + 2
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > M:
                break
            max_u = isqrt(M // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                if p <= M:
                    pw_set.add(p)
        return sorted(pw_set)

    # Limits
    # We want to search N up to ~10^10.
    # Let A be powerful numbers up to M_A = 10^10.
    # Let B be powerful numbers up to M_B = 10^7.
    # We compute sums a+b for a in A, b in B.
    # This covers N up to 10^10 + 10^7 ~ 10^10.
    # But we miss pairs where both a and b are large (e.g., 5*10^9 + 5*10^9).
    # However, the density of powerful numbers is low.
    # K(10^10) ~ 10^10 * C / log(10^10) ?
    # Number of powerful numbers <= x is ~ C * x^{1/2} / log(x)^{1/3} ? No.
    # It is ~ x^{1/2} * polylog?
    # Actually, count is ~ c * x^{1/2}.
    # For x=10^10, count ~ 10^5.
    # For x=10^7, count ~ 10^3.5 ~ 3000?
    # Let's estimate K.
    # K(10^7) was ~4500 in the prompt.
    # K(10^10) should be roughly sqrt(1000) times larger?
    # sqrt(10^10) = 10^5. sqrt(10^7) = 3162.
    # Ratio ~ 31.
    # So K_A ~ 4500 * 31 ~ 140,000.
    # K_B ~ 4500.
    # Product ~ 6 * 10^8.
    # This is too big for a single NumPy broadcast (600M elements * 8 bytes = 4.8GB).
    
    # We need to be smarter.
    # We can iterate over chunks of A.
    # Or reduce M_A.
    # Let's try M_A = 10^9, M_B = 10^8.
    # K(10^9) ~ sqrt(100) * K(10^7) ~ 10 * 4500 = 45,000.
    # K(10^8) ~ sqrt(10) * K(10^7) ~ 3.16 * 4500 = 14,000.
    # Product ~ 6 * 10^8. Still too big.
    
    # Let's try M_A = 10^8, M_B = 10^8.
    # K ~ 14,000 each.
    # Product ~ 2 * 10^8.
    # Array size 200M * 8 bytes = 1.6GB. Might be tight but possible on large machines.
    # But we can do better with chunking.
    
    # Strategy:
    # 1. Generate B (up to M_B = 10^8). K_B ~ 14,000.
    # 2. Generate A (up to M_A = 10^10). K_A ~ 140,000.
    # 3. Iterate A in chunks of size C_A.
    # 4. For each chunk, compute A_chunk + B using broadcasting.
    # 5. Use np.bincount to accumulate counts.
    
    M_A = 10_000_000_000  # 10^10
    M_B = 100_000_000     # 10^8
    
    # Generate B first (smaller)
    B_list = generate_powerful(M_B)
    B_arr = np.array(B_list, dtype=np.int64)
    K_B = len(B_arr)
    
    # Generate A
    A_list = generate_powerful(M_A)
    A_arr = np.array(A_list, dtype=np.int64)
    K_A = len(A_arr)
    
    # Check time
    if time.time() > deadline - 20:
        # Fallback to smaller search if generation was slow
        M_A = 1_000_000_000
        M_B = 10_000_000
        B_arr = np.array(generate_powerful(M_B), dtype=np.int64)
        A_arr = np.array(generate_powerful(M_A), dtype=np.int64)
        K_A = len(A_arr)
        K_B = len(B_arr)

    # We want to find N with max representations.
    # N = a + b.
    # We will accumulate counts in a dictionary or sparse array.
    # Max N is M_A + M_B.
    # Using a dictionary is safer for memory if N is large and sparse.
    # But np.bincount is very fast if we can map to indices.
    # However, N can be up to 10^10. An array of size 10^10 is impossible.
    # So we MUST use a dictionary or sort-based counting.
    
    # Alternative:
    # Compute all sums, sort them, then count frequencies.
    # Total sums ~ K_A * K_B.
    # If K_A=140k, K_B=14k, total ~ 2 billion.
    # Sorting 2 billion 64-bit integers is ~ 16GB memory and very slow.
    
    # We need to reduce the number of sums.
    # We only care about N that have MANY representations.
    # Most N have 0 or 1.
    # We can use a "probabilistic" filter or just accept that we might miss the global max if it's formed by two very large numbers not in our sets?
    # No, A covers up to 10^10. B covers up to 10^8.
    # Any pair (a,b) with a<=10^10, b<=10^8 is covered.
    # We miss pairs where b > 10^8.
    # But by symmetry, if we swap A and B, we cover pairs where a<=10^8, b<=10^10.
    # The union covers all pairs where min(a,b) <= 10^8.
    # What about pairs where both a,b > 10^8?
    # Such pairs sum to N > 2*10^8.
    # Are there N > 2*10^8 with many representations where both parts are large?
    # Likely, but less dense.
    # Let's assume the max count is found in the "mixed" region or lower N.
    # To be safe, we can do two passes:
    # Pass 1: A (up to 10^10) + B (up to 10^8).
    # Pass 2: A (up to 10^8) + B (up to 10^10). (This is symmetric, adds nothing new if we just sum counts).
    # Actually, the set of sums is the same.
    # We just need to ensure we don't double count?
    # No, we are counting pairs (a,b).
    # If we do Pass 1, we count all pairs where a in A_large, b in B_small.
    # If we do Pass 2, we count all pairs where a in A_small, b in B_large.
    # The union of these two sets of pairs is { (a,b) | a<=10^10, b<=10^8 OR a<=10^8, b<=10^10 }.
    # This misses { (a,b) | a>10^8, b>10^8 }.
    # Given K(10^8) ~ 14k, the number of such pairs is 14k^2 ~ 200M.
    # This is manageable!
    # So we can do:
    # 1. Count pairs where a <= M_small, b <= M_large. (Chunked)
    # 2. Count pairs where a > M_small, b > M_small. (Brute force or smaller chunked)
    
    M_small = 100_000_000  # 10^8
    M_large = 10_000_000_000 # 10^10
    
    # Generate lists
    P_small = np.array(generate_powerful(M_small), dtype=np.int64)
    P_large = np.array(generate_powerful(M_large), dtype=np.int64)
    
    # Split P_large into "small part" (<= M_small) and "large part" (> M_small)
    # P_small is exactly the small part.
    # P_large contains P_small.
    # Let's create P_large_only = P_large[P_large > M_small]
    
    mask_large = P_large > M_small
    P_large_only = P_large[mask_large]
    
    # Counts dictionary
    counts = {}
    
    # Helper to add sums to counts
    def add_sums(X, Y, counts_dict, chunk_size=10000):
        # X: array, Y: array
        # We iterate over X in chunks to avoid huge temporary arrays
        Kx = len(X)
        Ky = len(Y)
        # If Ky is large, we might want to chunk Y too?
        # Let's chunk X.
        for i in range(0, Kx, chunk_size):
            if time.time() > deadline - 10:
                return
            X_chunk = X[i:i+chunk_size]
            # Sums = X_chunk[:, None] + Y[None, :]
            # This creates a (chunk_size, Ky) array.
            # If chunk_size=10000, Ky=14000 -> 140M elements. 1.1GB. Too big.
            # Reduce chunk_size.
            # Let's target 10M elements per chunk.
            # If Ky=14000, chunk_size = 10M/14000 ~ 700.
            # Let's calculate dynamic chunk size.
            target_elements = 5_000_000
            cs = max(1, target_elements // Ky)
            
            for j in range(0, len(X_chunk), cs):
                X_sub = X_chunk[j:j+cs]
                sums = X_sub[:, None] + Y[None, :]
                # Flatten and count
                flat_sums = sums.ravel()
                # Use np.unique with return_counts
                unique_sums, unique_counts = np.unique(flat_sums, return_counts=True)
                for s, c in zip(unique_sums, unique_counts):
                    counts_dict[s] = counts_dict.get(s, 0) + c
                # Check time
                if time.time() > deadline - 10:
                    return

    # Pass 1: a in P_small, b in P_large
    # This covers all pairs where a <= M_small.
    # Note: This includes pairs where b <= M_small (both small) and b > M_small.
    add_sums(P_small, P_large, counts, chunk_size=1000)
    
    # Pass 2: a in P_large_only, b in P_small
    # This covers pairs where a > M_small and b <= M_small.
    # Note: This is symmetric to the subset of Pass 1 where b > M_small and a <= M_small?
    # No.
    # Pass 1 covers: (a<=Ms, b<=Ms) and (a<=Ms, b>Ms).
    # Pass 2 covers: (a>Ms, b<=Ms).
    # The union covers all pairs where min(a,b) <= Ms.
    # We still miss (a>Ms, b>Ms).
    
    if time.time() < deadline - 15:
        add_sums(P_large_only, P_small, counts, chunk_size=1000)
        
    # Pass 3: a in P_large_only, b in P_large_only
    # This covers the remaining pairs.
    # K_large_only ~ K_large - K_small ~ 140k - 14k = 126k.
    # 126k * 126k ~ 1.6 * 10^10. Too big.
    # We must skip this or use a very coarse approximation?
    # If we skip it, we might miss the best N if it is composed of two large numbers.
    # However, the density of powerful numbers decreases.
    # Let's check if we have time.
    # If not, we proceed with what we have.
    
    # Find best N
    best_N = 0
    best_count = 0
    if counts:
        # Get top candidates
        # Sorting 1M+ keys might be slow.
        # Let's just iterate and keep top 10?
        # Or use heapq.
        import heapq
        # We want max count.
        # Let's just find the max.
        # If counts is huge, this is O(N).
        for N, c in counts.items():
            if c > best_count:
                best_count = c
                best_N = N
                
    # Verify and extract pairs for best_N
    # We need to find pairs (a,b) such that a+b=best_N, a<=b, a,b powerful.
    # We can iterate through P_small and check if best_N - a is in P_large (set).
    # Or iterate through all powerful numbers up to best_N/2.
    # best_N can be up to 2*10^10.
    # We don't have a list of all powerful numbers up to 2*10^10.
    # But we have P_large up to 10^10.
    # If best_N <= 10^10, we can check against P_large.
    # If best_N > 10^10, one of the numbers must be > 10^10?
    # No, if a+b > 10^10, and a,b <= 10^10, it's possible.
    # But we only generated up to 10^10.
    # If best_N is found from sums of P_large, it is <= 2*10^10.
    # To verify, we need to check if a and b are powerful.
    # We can use the generation function to check? No, too slow.
    # We can check if a is in P_large_set and b is in P_large_set?
    # This only works if a,b <= 10^10.
    # If best_N > 10^10, and a,b <= 10^10, then both are in P_large.
    # So we can verify using P_large_set.
    
    P_large_set = set(P_large)
    
    pairs = []
    # Iterate a up to best_N // 2
    # We only have P_large.
    # If best_N is small, we might miss small a?
    # P_large contains all powerful numbers up to 10^10, including small ones.
    # So iterating P_large is sufficient IF best_N/2 <= 10^10.
    # If best_N > 2*10^10, we are in trouble, but our sums max out at 2*10^10.
    
    # Optimization: Only iterate a in P_large where a <= best_N // 2
    # P_large is sorted.
    # Find index of first element > best_N // 2
    import bisect
    limit_a = best_N // 2
    idx_end = bisect.bisect_right(P_large, limit_a)
    
    for i in range(idx_end):
        a = P_large[i]
        b = best_N - a
        if b in P_large_set:
            if a <= b:
                pairs.append([int(a), int(b)])
                
    # If no pairs found (e.g. best_N was 0 or verification failed), fallback
    if not pairs:
        # Fallback to a known valid small N
        # 2 = 1+1
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
