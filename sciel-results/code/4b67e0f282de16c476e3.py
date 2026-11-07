import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s
    
    def check_valid(elements):
        """Check if no two subsets of different cardinalities have the same sum."""
        n = len(elements)
        # For each cardinality k, compute the set of all subset sums of size k
        # Use DP: sums_by_k[k] = set of achievable sums with exactly k elements
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        
        for i in range(n):
            e = elements[i]
            # Update in reverse to avoid using element multiple times
            for k in range(i + 1, 0, -1):
                new_sums = set()
                for s in sums_by_k[k - 1]:
                    new_sums.add(s + e)
                sums_by_k[k].update(new_sums)
        
        # Check: for any k1 != k2, sums_by_k[k1] and sums_by_k[k2] must be disjoint
        for k1 in range(n + 1):
            for k2 in range(k1 + 1, n + 1):
                if sums_by_k[k1] & sums_by_k[k2]:
                    return False
        return True
    
    # Strategy: Try to build a large set using a superincreasing-like sequence
    # but allow some flexibility. Start with powers of 2 scaled, then try to add more.
    
    # First, try a simple superincreasing sequence: 1, 2, 4, 8, ...
    # This gives globally unique sums, hence valid.
    # Max in 1..256: 1,2,4,8,16,32,64,128 -> 8 elements
    # Can we do better?
    
    # Try: start with small numbers, but not necessarily powers of 2.
    # Use a greedy approach: try to add elements from a candidate pool.
    
    # Candidate: all numbers 1..256
    # Greedy: try to add elements in some order, checking validity.
    
    # Better: use a known construction. If we take elements a_i where a_i > sum(a_1..a_{i-1}),
    # then all subset sums are unique. Maximize count by choosing smallest such elements.
    # a_1 = 1, a_2 = 2, a_3 = 4, ..., a_8 = 128. Sum = 255. a_9 would need > 255, but max is 256.
    # a_9 = 256? 256 > 255, yes! So {1,2,4,8,16,32,64,128,256} is valid. 9 elements.
    
    base_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    if check_valid(base_set):
        best = list(base_set)
    else:
        best = list(range(1, 9))
    
    # Try to improve: can we replace some elements to fit more?
    # The superincreasing condition is sufficient but not necessary.
    # Let's try to find a larger set using search.
    
    # Try sets of size 10, 11, ... by modifying the base set.
    # Idea: if we drop 1 and use smaller gaps, maybe we can fit more.
    # But the constraint is about cross-cardinality sum separation.
    
    # Let's try a different approach: use elements that are all even (or all odd) 
    # to create separation. Actually, let's just try to find any valid set of size > 9.
    
    # Try: replace 256 with something smaller and add another element.
    # Or try a completely different construction.
    
    # Let's try: {2, 3, 5, 9, 17, 33, 65, 129, 257} - no, 257 > 256.
    # {2, 3, 5, 9, 17, 33, 65, 129} - 8 elements, sum = 263. 
    # Check if superincreasing: 2, 3>2, 5>5? No, 5 = 2+3. Not superincreasing.
    
    # Let's just do a local search: start with a valid set, try to add elements from 1..256 not in the set.
    
    def try_add_element(current_set, candidate):
        """Try to add candidate to current_set and check validity."""
        new_set = list(current_set) + [candidate]
        new_set.sort()
        if check_valid(new_set):
            return new_set
        return None
    
    # Try to add elements to base_set
    current = list(base_set)
    improved = True
    while improved:
        improved = False
        candidates = [x for x in range(1, 257) if x not in current]
        # Try candidates in increasing order (smaller elements are more likely to fit)
        for c in candidates:
            if time.time() > deadline:
                break
            new_set = try_add_element(current, c)
            if new_set is not None:
                current = new_set
                improved = True
                break
        if time.time() > deadline:
            break
    
    if len(current) > len(best):
        best = current
    
    # Try starting from different initial sets
    # Try sets with smaller elements to see if we can fit more
    for start_val in [1, 2, 3]:
        if time.time() > deadline:
            break
        # Build a superincreasing-like set starting from start_val
        s = []
        total = 0
        for x in range(start_val, 257):
            if x > total and (not s or x > s[-1]):
                s.append(x)
                total += x
                if len(s) >= len(best) + 1:
                    break
        if len(s) > len(best) and check_valid(s):
            best = s
    
    # Final verification
    if not check_valid(best):
        # Fallback to a known valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
