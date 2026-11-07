import time
from itertools import combinations

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    Strategy:
    1. Start with the known 31-element AP [226..256].
    2. Systematically search for larger valid sets (32+) using:
       a. Dense APs with various parameters
       b. Local search / perturbation of the base set
       c. Structured sets (small + large clusters)
    3. Use exact cardinality-indexed subset-sum range checking for validation.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Known valid 31-element solution
    fallback = list(range(226, 257))
    
    def is_valid_fast(s):
        """
        Fast validation: check that for all c1 < c2, max_sum(c1) < min_sum(c2).
        For a sorted set, min_sum(c) = sum of c smallest, max_sum(c) = sum of c largest.
        This is a sufficient condition (if ranges don't overlap, no sum collision across cardinalities).
        """
        n = len(s)
        if n == 0:
            return True
        s_sorted = sorted(s)
        # Prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        # min_sum[c] = prefix[c], max_sum[c] = prefix[n] - prefix[n-c]
        min_sums = [prefix[c] for c in range(n + 1)]
        max_sums = [prefix[n] - prefix[n - c] for c in range(n + 1)]
        # Check: for all c1 < c2, max_sums[c1] < min_sums[c2]
        # It suffices to check adjacent cardinalities: max_sums[c] < min_sums[c+1]
        # because if adjacent ranges are disjoint, all ranges are disjoint (by transitivity)
        for c in range(n):
            if max_sums[c] >= min_sums[c + 1]:
                return False
        return True
    
    def is_valid_exact(s):
        """
        Exact validation using subset-sum DP per cardinality.
        For each cardinality c, compute the set of all achievable sums.
        Then check that no sum appears in two different cardinality sets.
        """
        n = len(s)
        if n == 0:
            return True
        # For each cardinality c, track which sums are achievable
        # sums_by_card[c] = set of achievable sums using exactly c elements
        sums_by_card = [set() for _ in range(n + 1)]
        sums_by_card[0].add(0)
        
        for x in s:
            # Process in reverse to avoid using x multiple times
            for c in range(n - 1, -1, -1):
                if sums_by_card[c]:
                    for sm in sums_by_card[c]:
                        sums_by_card[c + 1].add(sm + x)
        
        # Check for collisions across cardinalities
        for c1 in range(n + 1):
            for c2 in range(c1 + 1, n + 1):
                if sums_by_card[c1] & sums_by_card[c2]:
                    return False
        return True
    
    best_size = 31
    best_set = fallback
    
    # Phase 1: Try dense APs of length 32+
    for n in range(32, 45):
        if time.time() > deadline:
            break
        for d in range(1, 5):
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            # Try several starting points
            candidates_a = set()
            candidates_a.add(1)
            candidates_a.add(max_a)
            for offset in range(0, min(10, max_a)):
                candidates_a.add(max_a - offset)
            candidates_a.add(max_a // 2)
            candidates_a.add(max_a - max_a // 2)
            
            for a in sorted(candidates_a):
                if a < 1 or a > max_a:
                    continue
                s = [a + i * d for i in range(n)]
                if s[-1] > 256:
                    continue
                if is_valid_fast(s):
                    if n > best_size:
                        best_size = n
                        best_set = s
                    break
    
    # Phase 2: Local search - perturb the base 31-element set to try to add elements
    if best_size < 32 and time.time() < deadline - 10:
        # Try contiguous ranges of length 32
        for a in range(1, 226):
            if time.time() > deadline:
                break
            s = list(range(a, a + 32))
            if s[-1] > 256:
                break
            if is_valid_fast(s):
                best_size = 32
                best_set = s
                break
        
        # Try non-contiguous: small cluster + large cluster
        if best_size < 32:
            for n_small in range(5, 28):
                if time.time() > deadline:
                    break
                n_large = 32 - n_small
                if n_large < 5:
                    continue
                s_small = list(range(1, n_small + 1))
                s_large = list(range(257 - n_large, 257))
                s = s_small + s_large
                if len(set(s)) == 32 and is_valid_fast(s):
                    best_size = 32
                    best_set = s
                    break
    
    # Phase 3: More sophisticated search for 32+ elements
    if best_size < 33 and time.time() < deadline - 5:
        # Try: take a 31-element AP and try to insert one more element
        # or modify elements to create room
        # Strategy: use a non-uniform spacing pattern
        # For n elements, we need: for all c < n, sum of c largest < sum of (c+1) smallest
        # This is equivalent to: for all c, sum(x_{n-c+1}..x_n) < sum(x_1..x_{c+1})
        
        # Try geometric-like progressions
        for base in [2, 3]:
            if time.time() > deadline:
                break
            candidates = []
            v = 1
            while v <= 256:
                candidates.append(v)
                v = v * base + 1
            if len(candidates) >= 32:
                # Try various subsets of size 32
                # First 32
                s = candidates[:32]
                if is_valid_fast(s):
                    best_size = 32
                    best_set = s
                # Last 32
                s = candidates[-32:]
                if is_valid_fast(s):
                    best_size = 32
                    best_set = s
                # Mix: first 16 + last 16
                s = candidates[:16] + candidates[-16:]
                if len(set(s)) == 32 and is_valid_fast(s):
                    best_size = 32
                    best_set = s
    
    # Phase 4: If we found a candidate with fast validation, do exact validation
    if best_size > 31 and time.time() < deadline - 2:
        if not is_valid_exact(best_set):
            # Fast validation was a false positive; fall back
            best_size = 31
            best_set = fallback
    
    # Final safety check
    if not is_valid_fast(best_set):
        return fallback
    
    # Ensure return type is correct
    result = [int(x) for x in best_set]
    return result
