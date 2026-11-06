import time
import itertools

# Precompute divisors of 5040 = 2^4 * 3^2 * 5 * 7
_DIVS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20,
    21, 24, 28, 30, 35, 36, 40, 42, 45, 48, 56, 60, 63, 70,
    72, 80, 84, 90, 105, 112, 120, 126, 140, 144, 168, 180,
    210, 224, 240, 252, 280, 315, 336, 360, 420, 504, 560,
    630, 720, 840, 1008, 1260, 1680, 2520, 5040
)
_DIV_SET = frozenset(_DIVS)
TARGET = 4979

def find_min_divisors(target, divs, max_terms):
    """
    Find the minimum number of distinct divisors from `divs` that sum to `target`.
    Returns the list of divisors if found, else None.
    Uses DFS with pruning.
    """
    divs = sorted(divs, reverse=True)
    n = len(divs)
    best = None
    
    # Check for solutions of size k from 1 to max_terms
    for k in range(1, max_terms + 1):
        # Use DFS to find a combination of k divisors summing to target
        # To optimize, we can use a set for O(1) lookups if we fix the first k-1
        # But for small k (3 or 4), simple iteration or meet-in-middle is fine.
        
        # For k=1:
        if k == 1:
            if target in _DIV_SET:
                return [target]
            continue
        
        # For k=2:
        if k == 2:
            found = False
            for d in divs:
                if d > target:
                    continue
                comp = target - d
                if comp in _DIV_SET and comp != d:
                    return [d, comp]
            continue
            
        # For k=3:
        if k == 3:
            # Iterate first two, check third
            # Optimization: sort divs descending. 
            # If d1 + d2 + d3 = target, and d1 >= d2 >= d3
            # We can iterate d1, d2 such that d1+d2 < target
            # and check if target-d1-d2 is in set and < d2
            for i in range(n):
                d1 = divs[i]
                if d1 > target:
                    continue
                rem1 = target - d1
                # d2 must be < d1 (distinct, and we can assume order to avoid duplicates)
                # Actually, distinct means values are different.
                # We just need to find any 3 distinct values.
                # To avoid permutations, we can enforce index order i < j < k
                # But divs is sorted by value. 
                # Let's just use a set for the third one.
                # Iterate d1, d2
                for j in range(n):
                    if i == j: continue
                    d2 = divs[j]
                    if d1 + d2 > target:
                        continue
                    comp = target - d1 - d2
                    if comp in _DIV_SET and comp != d1 and comp != d2:
                        return [d1, d2, comp]
            continue

        # For k=4:
        if k == 4:
            # This is heavier. 
            # Meet in the middle: split divs into two halves?
            # Or iterate 2 pairs.
            # Number of divs is 60. Pairs is ~1800.
            # We can generate all pairs (d1, d2) with d1 < d2, sum = s.
            # Store sums in a dict: sum -> (d1, d2)
            # Then iterate through pairs again, check if target - sum is in dict.
            # Ensure all 4 are distinct.
            
            pair_sums = {}
            # To ensure distinctness, we can just store one pair per sum.
            # If we find a collision, we might need another pair.
            # But for existence check, one pair per sum is enough if we can find a disjoint pair.
            # Actually, we need to check if the two pairs are disjoint.
            # So we should store all pairs for a sum? Or just one and backtrack?
            # Given the small size, let's just iterate 4 nested loops with pruning?
            # 60^4 is 12.9M. Might be too slow in Python if not pruned.
            # Pruning: d1 > d2 > d3 > d4.
            # Sum of largest 4: 5040+2520+1680+1260 = 10500 > 4979.
            # So we can prune.
            
            # Let's try the pair sum approach.
            # Generate all pairs (i, j) with i < j.
            # There are 60*59/2 = 1770 pairs.
            # Store in dict: sum -> list of (i, j)
            # But we need values.
            
            sum_to_pairs = {}
            for i in range(n):
                d1 = divs[i]
                if d1 > target:
                    continue
                for j in range(i + 1, n):
                    d2 = divs[j]
                    s = d1 + d2
                    if s > target:
                        continue
                    if s not in sum_to_pairs:
                        sum_to_pairs[s] = []
                    sum_to_pairs[s].append((d1, d2))
            
            # Now for each pair (d3, d4) with sum s2, check if target - s2 is in sum_to_pairs
            # and if the pairs are disjoint.
            for i in range(n):
                d3 = divs[i]
                if d3 > target:
                    continue
                for j in range(i + 1, n):
                    d4 = divs[j]
                    s2 = d3 + d4
                    if s2 >= target:
                        continue
                    rem = target - s2
                    if rem in sum_to_pairs:
                        for (d1, d2) in sum_to_pairs[rem]:
                            if d1 != d3 and d1 != d4 and d2 != d3 and d2 != d4:
                                return [d1, d2, d3, d4]
            continue
            
    return None

def construct():
    # Check for solutions with fewer than 5 divisors
    # We know 5 works.
    # Check 4
    sol = find_min_divisors(TARGET, _DIVS, 4)
    if sol:
        return sol
    
    # If no 4-divisor solution, return the known 5-divisor solution
    # (2520, 1680, 560, 210, 9)
    return [2520, 1680, 560, 210, 9]
