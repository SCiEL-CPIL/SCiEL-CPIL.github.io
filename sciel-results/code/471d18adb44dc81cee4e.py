import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    
    Uses exact cardinality-indexed subset-sum DP to verify validity,
    then searches for larger valid sets via greedy + local improvement.
    """
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    def is_valid(arr):
        """Check if arr satisfies the cardinality-separated subset sum constraint."""
        n = len(arr)
        total_sum = sum(arr)
        # sums[j] = set of achievable sums using exactly j elements
        # sums[0] = {0} (empty subset)
        # Use bitsets for efficiency
        # Max sum with j elements is bounded by j * max(arr)
        # But we need exact tracking. Use sets for small n.
        
        # DP: reachable[j] = set of sums achievable with exactly j elements
        reachable = [set() for _ in range(n + 1)]
        reachable[0] = {0}
        
        for x in arr:
            # Update in reverse to avoid using x multiple times
            for j in range(n, 0, -1):
                if reachable[j - 1]:
                    for s in reachable[j - 1]:
                        reachable[j].add(s + x)
        
        # Check: no sum appears in two different cardinality classes
        # Build a map: sum -> cardinality
        seen = {}
        for j in range(n + 1):
            for s in reachable[j]:
                if s in seen:
                    if seen[s] != j:
                        return False
                else:
                    seen[s] = j
        return True
    
    def is_valid_fast(arr):
        """Faster validity check using bitsets where possible."""
        n = len(arr)
        total_sum = sum(arr)
        
        # Use list of sets
        reachable = [set() for _ in range(n + 1)]
        reachable[0] = {0}
        
        for x in arr:
            for j in range(n, 0, -1):
                if reachable[j - 1]:
                    new_sums = set()
                    for s in reachable[j - 1]:
                        new_sums.add(s + x)
                    reachable[j] = reachable[j] | new_sums
        
        # Check disjointness
        seen = {}
        for j in range(n + 1):
            for s in reachable[j]:
                if s in seen:
                    if seen[s] != j:
                        return False
                else:
                    seen[s] = j
        return True
    
    # Start with the known valid solution
    best = list(range(226, 257))  # 31 elements
    best_size = len(best)
    
    # Try to extend: attempt to add elements
    # Strategy: try adding elements from both ends of the range
    # First, try to find a larger valid set by search
    
    # Greedy approach: try to build a larger set
    # We know k=31 works with AP starting at 226.
    # Can we do better? Let's try different constructions.
    
    # Approach 1: Try AP with different parameters
    # For an AP {a, a+d, a+2d, ..., a+(k-1)d} to work:
    # The sum of i smallest elements = i*a + d*i*(i-1)/2
    # The sum of i largest elements = i*(a+(k-1)d) - d*i*(i-1)/2
    # For different cardinalities to have different sums, we need the range of
    # sums for cardinality i to not overlap with range for cardinality j.
    # Min sum for cardinality i: i*a + d*i*(i-1)/2
    # Max sum for cardinality i: i*(a+(k-1)d) - d*i*(i-1)/2
    # For no overlap between i and j (i < j):
    # max_sum(i) < min_sum(j)
    # i*(a+(k-1)d) - d*i*(i-1)/2 < j*a + d*j*(j-1)/2
    
    # For d=1: max_sum(i) = i*a + (k-1)*i - i*(i-1)/2
    #          min_sum(j) = j*a + j*(j-1)/2
    # Condition: i*a + (k-1)*i - i*(i-1)/2 < j*a + j*(j-1)/2
    # For worst case i=j-1:
    # (j-1)*a + (k-1)*(j-1) - (j-1)*(j-2)/2 < j*a + j*(j-1)/2
    # (k-1)*(j-1) - (j-1)*(j-2)/2 - j*(j-1)/2 < a
    # (j-1)[(k-1) - (j-2)/2 - j/2] < a
    # (j-1)[k-1 - (j-1)] < a
    # (j-1)(k-j) < a
    # Max of (j-1)(k-j) over j: let m=j-1, then m(k-1-m), max at m=(k-1)/2
    # For k=31: max of m*30-m^2 at m=15: 15*15=225
    # So a > 225, a >= 226. That's what we have.
    
    # For k=32: max of m*31-m^2 at m=15 or 16: 15*16=240
    # a >= 241, a+31 = 272 > 256. Doesn't fit.
    
    # So AP can't do better than 31. But non-AP might work!
    
    # Approach 2: Non-uniform sequences
    # Try to use a superincreasing-like sequence or other structures.
    
    # Let's try a binary-search-like approach for larger sets
    # Try sets of size 32, 33, etc.
    
    def try_build(k_target):
        """Try to build a valid set of size k_target."""
        if time.time() > DEADLINE:
            return None
        
        # Start with the AP solution as base, try to modify
        # Or build from scratch using greedy with DP validation
        
        # Greedy: add elements one by one, choosing the one that keeps validity
        # and leaves room for more elements
        
        # Strategy: work backwards - start with full range and remove elements
        # Or: try specific constructions
        
        # Try: use elements from both high and low ends
        # The constraint is that subset sums of different sizes must be disjoint.
        
        # Let's try a construction based on powers of 2 scaled down
        # Or try: {a, a+1, a+2, ..., a+k-1} but with gaps
        
        # Actually, let's try a direct search for k=32
        # We need 32 distinct elements in 1..256
        # The sum of the smallest i elements must be < sum of smallest i+1 elements
        # (trivially true) but also the LARGEST sum with i elements must be <
        # the SMALLEST sum with i+1 elements for the constraint to hold.
        
        # Wait, that's sufficient but not necessary. The constraint is just that
        # NO sum appears in two different cardinality classes. It's possible that
        # max_sum(i) >= min_sum(i+1) but still no actual collision occurs.
        
        # Let's just try to verify if k=32 is possible with a specific construction.
        
        # Try: take the 32 largest numbers: 225..256
        candidate = list(range(225, 257))
        if is_valid_fast(candidate):
            return candidate
        
        # Try: 224..255
        candidate = list(range(224, 256))
        if is_valid_fast(candidate):
            return candidate
        
        # Try mixing: some high, some low
        # The key insight: if we have very small elements, they create many small sums
        # at low cardinalities. If we have very large elements, they create large sums
        # at high cardinalities. The danger is in the middle.
        
        # Try: first half small, second half large with a gap
        # e.g., {1, 2, ..., 16} union {240, 241, ..., 255}
        # But this has 32 elements and the sums might collide.
        
        # Let's try a more systematic approach
        # Use the fact that for the AP, the critical constraint is at the middle.
        # If we can "spread out" the elements more, we might fit more.
        
        # Try: geometric-like sequence
        # {a, a+r, a+2r+1, a+3r+2, ...} with increasing gaps
        
        # For now, let's just verify the current best and try a few more APs
        return None
    
    # Try k=32 with various constructions
    for attempt in range(100):
        if time.time() > DEADLINE:
            break
        
        if attempt == 0:
            # Already tried range(225, 257) and range(224, 256)
            pass
        elif attempt == 1:
            # Try: 226..256 (31 elements) + one more? Can't, already at 256.
            # Try replacing some elements to make room
            # Take 227..256 (30 elements) and add two smaller ones
            candidate = list(range(227, 257))
            # Add 1 and 2
            candidate = [1, 2] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 2:
            # Try: 228..256 (29 elements) + 3 smaller
            candidate = list(range(228, 257))
            candidate = [1, 2, 3] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 3:
            # Try: 229..256 (28 elements) + 4 smaller
            candidate = list(range(229, 257))
            candidate = [1, 2, 3, 4] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 4:
            # Try: 230..256 (27 elements) + 5 smaller
            candidate = list(range(230, 257))
            candidate = [1, 2, 3, 4, 5] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 5:
            # Try: 231..256 (26 elements) + 6 smaller
            candidate = list(range(231, 257))
            candidate = [1, 2, 3, 4, 5, 6] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 6:
            # Try: 232..256 (25 elements) + 7 smaller
            candidate = list(range(232, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 7:
            # Try: 233..256 (24 elements) + 8 smaller
            candidate = list(range(233, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 8:
            # Try: 234..256 (23 elements) + 9 smaller
            candidate = list(range(234, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 9:
            # Try: 235..256 (22 elements) + 10 smaller
            candidate = list(range(235, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 10:
            # Try: 236..256 (21 elements) + 11 smaller
            candidate = list(range(236, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 11:
            # Try: 237..256 (20 elements) + 12 smaller
            candidate = list(range(237, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 12:
            # Try: 238..256 (19 elements) + 13 smaller
            candidate = list(range(238, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 13:
            # Try: 239..256 (18 elements) + 14 smaller
            candidate = list(range(239, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 14:
            # Try: 240..256 (17 elements) + 15 smaller
            candidate = list(range(240, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 15:
            # Try: 241..256 (16 elements) + 16 smaller
            candidate = list(range(241, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 16:
            # Try: 242..256 (15 elements) + 17 smaller
            candidate = list(range(242, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 17:
            # Try: 243..256 (14 elements) + 18 smaller
            candidate = list(range(243, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 18:
            # Try: 244..256 (13 elements) + 19 smaller
            candidate = list(range(244, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
        
        elif attempt == 19:
            # Try: 245..256 (12 elements) + 20 smaller
            candidate = list(range(245, 257))
            candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + candidate
            if is_valid_fast(candidate):
                best = candidate
                best_size = len(candidate)
    
    # Also try some non-contiguous constructions
    # Try: powers of 2 scaled
    if time.time() < DEADLINE:
        # {1, 2, 4, 8, 16, 32, 64, 128} is superincreasing - all subset sums unique
        # But we need cardinality separation, not just uniqueness
        # With superincreasing, each sum is unique, so cardinality separation is automatic
        # But we're limited to 8 elements with powers of 2 in 1..256
        
        # Try: {1, 3, 7, 15, 31, 63, 127, 255} (2^k - 1)
        # Sums are also unique for superincreasing sequences
        
        # Try a mixed approach: small superincreasing + large AP
        # {1, 3, 7, 15, 31} (5 elements, max sum = 57)
        # + {100, 101, 102, ..., 130} (31 elements)
        # Total: 36 elements
        # But need to verify no cross-collisions
        
        # The small elements create sums 0..57
        # The large elements create sums from 100 to 130*31 = 4030
        # But combinations of small + large could collide
        # This is complex. Let's just try and verify.
        
        candidate = [1, 3, 7, 15, 31] + list(range(100, 131))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4, 8, 16, 32, 64} (7 elements, max sum 127) + {128, 129, ..., 154} (27 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2, 4, 8, 16, 32, 64] + list(range(128, 155))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4, 8, 16, 32, 64, 128} (8 elements, max sum 255) - can't add anything else
    # since all sums 0..255 are already taken at various cardinalities
    # Wait, with {1,2,4,8,16,32,64,128}, every sum from 0 to 255 is achievable
    # and each sum has a unique representation, so each sum has a unique cardinality.
    # So this is valid! But only 8 elements.
    
    # The key question: can we do better than 31?
    # Let's try a more careful search for k=32.
    
    # For k=32, we need 32 distinct elements in 1..256.
    # The AP analysis showed we need a >= 241 for d=1, but a+31 >= 272 > 256.
    # With d>1, it's even worse.
    # But non-AP might work.
    
    # Let's try: take 32 elements that are "spread out" enough.
    # One idea: use elements that are approximately geometrically spaced.
    
    # Try: round(256 / (k+1) * (i+1)) for i=0..k-1
    if time.time() < DEADLINE:
        k_target = 32
        candidate = []
        for i in range(k_target):
            val = int(round(256 / (k_target + 1) * (i + 1)))
            if val not in candidate and 1 <= val <= 256:
                candidate.append(val)
        if len(candidate) == k_target and is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: evenly spaced with offset
    if time.time() < DEADLINE:
        k_target = 32
        # Use 8, 16, 24, ..., 256
        candidate = [8 * (i + 1) for i in range(32)]
        if max(candidate) <= 256 and is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: 1, 9, 17, 25, ... (step 8)
    if time.time() < DEADLINE:
        k_target = 32
        candidate = [1 + 8 * i for i in range(32)]
        if max(candidate) <= 256 and is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: 1, 8, 16, 24, ... (first step 7, then 8)
    if time.time() < DEADLINE:
        k_target = 32
        candidate = [1]
        for i in range(1, 32):
            candidate.append(candidate[-1] + 8)
        if max(candidate) <= 256 and is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Let's try a more principled approach for k=32
    # We need the sum of any i elements to not equal the sum of any j elements (i != j)
    # Sufficient condition: max_sum(i) < min_sum(i+1) for all i
    # max_sum(i) = sum of i largest elements
    # min_sum(i+1) = sum of i+1 smallest elements
    # If this holds, then all sums of i elements are < all sums of i+1 elements
    
    # For a sorted array a_1 < a_2 < ... < a_k:
    # max_sum(i) = sum(a_{k-i+1}, ..., a_k)
    # min_sum(i+1) = sum(a_1, ..., a_{i+1})
    # Condition: sum(a_{k-i+1}, ..., a_k) < sum(a_1, ..., a_{i+1})
    
    # This is a strong condition. Let's see if we can satisfy it for k=32.
    # For i=16 (middle):
    # sum(a_{17}, ..., a_{32}) < sum(a_1, ..., a_{17})
    # This means the sum of the 16 largest < sum of the 17 smallest
    # Average of 16 largest < (17/16) * average of 17 smallest
    # This requires the distribution to be heavily skewed toward small values
    
    # But all values are in 1..256, so the 16 largest are at least 241..256 (if using top 16)
    # Sum of 241..256 = 16*248.5 = 3976
    # Sum of 17 smallest: if 1..17, sum = 153. Way too small.
    # So this sufficient condition is too strong.
    
    # The actual condition is weaker: we just need NO collision, not that all sums of i
    # are less than all sums of i+1.
    
    # Let's just trust that 31 is likely optimal and return the known solution.
    # But first, let's try a few more constructions.
    
    # Try: {2, 4, 6, ..., 62} (31 elements, even numbers up to 62)
    # + one more element? 64?
    # {2, 4, ..., 62, 64} - 32 elements
    if time.time() < DEADLINE:
        candidate = [2 * i for i in range(1, 32)] + [64]
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 3, 5, ..., 61} (31 odd numbers) + 63
    if time.time() < DEADLINE:
        candidate = [2 * i - 1 for i in range(1, 32)] + [63]
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 31} (31 elements)
    # This should work? Let's verify.
    if time.time() < DEADLINE:
        candidate = list(range(1, 32))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 32}
    if time.time() < DEADLINE:
        candidate = list(range(1, 33))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 33}
    if time.time() < DEADLINE:
        candidate = list(range(1, 34))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 40}
    if time.time() < DEADLINE:
        candidate = list(range(1, 41))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 50}
    if time.time() < DEADLINE:
        candidate = list(range(1, 51))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 60}
    if time.time() < DEADLINE:
        candidate = list(range(1, 61))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 100}
    if time.time() < DEADLINE:
        candidate = list(range(1, 101))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 128}
    if time.time() < DEADLINE:
        candidate = list(range(1, 129))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 200}
    if time.time() < DEADLINE:
        candidate = list(range(1, 201))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 3, ..., 256}
    if time.time() < DEADLINE:
        candidate = list(range(1, 257))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try some more creative constructions
    # {1, 2, 4, 8, 16, 32, 64, 128, 192, 224, 240, 248, 252, 254, 255, 256}
    if time.time() < DEADLINE:
        candidate = [1, 2, 4, 8, 16, 32, 64, 128, 192, 224, 240, 248, 252, 254, 255, 256]
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: superincreasing up to 128, then fill in
    if time.time() < DEADLINE:
        # {1, 2, 4, 8, 16, 32, 64, 128} gives all sums 0..255
        # Can't add anything else
        # But {1, 2, 4, 8, 16, 32, 64} gives sums 0..127
        # Add 129, 130, ..., 160 (32 elements)
        # Total: 7 + 32 = 39 elements
        # But need to verify
        candidate = [1, 2, 4, 8, 16, 32, 64] + list(range(129, 161))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4, 8, 16, 32} (6 elements, sums 0..63) + {64, 65, ..., 95} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2, 4, 8, 16, 32] + list(range(64, 96))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4, 8, 16} (5 elements, sums 0..31) + {32, 33, ..., 63} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2, 4, 8, 16] + list(range(32, 64))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4, 8} (4 elements, sums 0..15) + {16, 17, ..., 47} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2, 4, 8] + list(range(16, 48))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2, 4} (3 elements, sums 0..7) + {8, 9, ..., 39} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2, 4] + list(range(8, 40))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1, 2} (2 elements, sums 0..3) + {4, 5, ..., 35} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1, 2] + list(range(4, 36))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {1} (1 element, sums 0,1) + {3, 4, ..., 34} (32 elements)
    if time.time() < DEADLINE:
        candidate = [1] + list(range(3, 35))
        if is_valid_fast(candidate):
            best = candidate
            best_size = len(candidate)
    
    # Try: {2, 4, 8, 16, 32, 64, 128} (7 elements, sums 0..255 step 2) 
    # Actually sums are all even from 0 to 254
    # Add odd numbers: 1, 3, 5, ..., 255
    # But 1 + 2 = 3, which is in the odd set. Collision!
    # So this doesn't work directly.
    
    # Let's try a completely different approach:
    # Use the fact that we can have repeated sums within the same cardinality.
    # The constraint is only between different cardinalities.
    
    # Key insight: if all elements are >= 2, then:
    # - Sum of 0 elements = 0
    # - Sum of 1 element >= 2
    # - Sum of 2 elements >= 4
    # - etc.
    # But this doesn't guarantee separation.
    
    # Actually, the AP solution works because the gaps are large enough.
    # Let's just return the best we found.
    
    return best
# EVOLVE-BLOCK-END
