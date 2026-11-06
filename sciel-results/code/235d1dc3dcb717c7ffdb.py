"""Rewritten program: targeted search for N with many powerful-number sum representations."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def remaining():
        return deadline - time.time()

    # Generate powerful numbers up to L using a^2 * b^3 form (b squarefree)
    def gen_powerful(L):
        max_b = int(round(L ** (1/3))) + 3
        # Sieve for squarefree numbers up to max_b
        is_sqfree = bytearray([1]) * (max_b + 1)
        if max_b >= 0:
            is_sqfree[0] = 0
        p = 2
        while p * p <= max_b:
            if is_sqfree[p]:
                p2 = p * p
                for j in range(p2, max_b + 1, p2):
                    is_sqfree[j] = 0
            p += 1
        
        sqfree = [i for i in range(1, max_b + 1) if is_sqfree[i]]
        
        powerful = set()
        for b in sqfree:
            b3 = b ** 3
            if b3 > L:
                break
            max_a = int(math.isqrt(L // b3))
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful.add(n)
        return sorted(powerful)

    # Strategy: 
    # 1. Generate powerful numbers up to a moderate bound (say 2e7)
    # 2. Compute all pairwise sums and find the N with the most representations
    # 3. Then try to extend: for that N, check if there are additional powerful pairs
    #    where one or both exceed our generation bound, by checking N-a directly
    
    # Step 1: Generate powerful numbers
    L = 25_000_000
    powerful = gen_powerful(L)
    n_pw = len(powerful)
    print(f"Generated {n_pw} powerful numbers up to {L}")
    
    if remaining() < 80:
        # Fallback: use smaller set
        L = 10_000_000
        powerful = gen_powerful(L)
        n_pw = len(powerful)
        print(f"Fallback: {n_pw} powerful numbers up to {L}")
    
    pw_arr = np.array(powerful, dtype=np.int64)
    pw_set = set(powerful)
    
    # Step 2: Find N with most representations using pairwise sums
    # Use chunked approach to manage memory
    best_N = 2
    best_count = 0
    best_pairs_list = []
    
    # Count sums using a dictionary (memory efficient for moderate n)
    # For n_pw up to ~15000, n_pw^2/2 ~ 112M which is too many for pure Python dict
    # Use numpy unique in chunks
    
    counts = defaultdict(int)
    chunk = 3000
    
    for i_s in range(0, n_pw, chunk):
        i_e = min(i_s + chunk, n_pw)
        pw_i = pw_arr[i_s:i_e]
        
        for j_s in range(i_s, n_pw, chunk):
            if remaining() < 30:
                break
            j_e = min(j_s + chunk, n_pw)
            pw_j = pw_arr[j_s:j_e]
            
            # Only pairs where pw_i <= pw_j (to ensure a <= b)
            # Since i_s <= j_s, most pairs satisfy this, but need to handle overlap
            sums = pw_i[:, None] + pw_j[None, :]
            mask = pw_i[:, None] <= pw_j[None, :]
            sums = sums[mask]
            
            if sums.size == 0:
                continue
            
            unique_sums, cnts = np.unique(sums, return_counts=True)
            for s, c in zip(unique_sums, cnts):
                counts[int(s)] += int(c)
        
        if remaining() < 30:
            break
    
    # Find best N from counts
    if counts:
        for n_val, cnt in counts.items():
            if cnt > best_count:
                best_count = cnt
                best_N = n_val
    
    print(f"Best N from pairwise sums: {best_N} with {best_count} pairs")
    
    # Step 3: Try to find more representations for best_N
    # Check if N-a is powerful for a in powerful where N-a > L (beyond our generation)
    # We need an is_powerful check for numbers up to N
    # N is at most ~50M (sum of two numbers up to 25M), so N-a is at most 25M
    # Actually all a,b in our set are <= L, so N <= 2L = 50M
    # So N-a <= 50M, which we could check by generating powerful up to 2L
    
    # Actually, let's also try: generate powerful up to 2*L and recheck
    # But that might be too slow. Instead, let's try a different N search.
    
    # Alternative: try N values that are "round" or have structure
    # For now, let's also try searching for N in a wider range by checking
    # specific candidate N values
    
    # Let's also try: for the best_N found, verify and collect pairs
    pairs = []
    for a in powerful:
        if a > best_N:
            break
        b = best_N - a
        if b >= a and b in pw_set:
            pairs.append([a, b])
    
    # Try to find additional pairs where b > L (not in our generated set)
    # We need to check if b is powerful. Since b <= best_N <= 2L, and we only
    # generated up to L, we might miss some. Let's generate up to 2L if time permits.
    
    if remaining() > 40 and best_N > L:
        # Generate powerful up to best_N (or 2L)
        L2 = min(best_N, 60_000_000)
        if L2 > L:
            powerful2 = gen_powerful(L2)
            pw_set2 = set(powerful2)
            pairs2 = []
            for a in powerful2:
                if a > best_N:
                    break
                b = best_N - a
                if b >= a and b in pw_set2:
                    pairs2.append([a, b])
            if len(pairs2) > len(pairs):
                pairs = pairs2
                best_count = len(pairs)
    
    # Also try: search for better N by checking N candidates from the top counts
    # Sort counts by value and try top candidates
    if remaining() > 50:
        top_candidates = sorted(counts.items(), key=lambda x: -x[1])[:20]
        for n_val, cnt in top_candidates:
            if n_val == best_N:
                continue
            if remaining() < 40:
                break
            # Verify count for this N
            pairs_c = []
            for a in powerful:
                if a > n_val:
                    break
                b = n_val - a
                if b >= a and b in pw_set:
                    pairs_c.append([a, b])
            if len(pairs_c) > best_count:
                best_count = len(pairs_c)
                best_N = n_val
                pairs = pairs_c
    
    # Ensure we have at least one valid pair
    if not pairs:
        # Fallback: find any valid pair
        for a in powerful:
            b = powerful[-1] - a if powerful else 1
            if a + b == powerful[-1] and b >= a and b in pw_set:
                pairs = [[a, b]]
                best_N = a + b
                break
        if not pairs and n_pw >= 2:
            pairs = [[powerful[0], powerful[1]]]
            best_N = powerful[0] + powerful[1]
        elif n_pw >= 1:
            pairs = [[powerful[0], 1]] if 1 in pw_set else [[1, 1]]
            best_N = 2
        else:
            pairs = [[1, 1]]
            best_N = 2
    
    # Final verification and cleanup
    # Remove duplicates and ensure a <= b
    pairs = sorted(set(tuple(p) for p in pairs))
    pairs = [[int(a), int(b)] for a, b in pairs if a <= b and a + b == best_N]
    best_count = len(pairs)
    
    print(f"Final: N={best_N}, {best_count} pairs")
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
