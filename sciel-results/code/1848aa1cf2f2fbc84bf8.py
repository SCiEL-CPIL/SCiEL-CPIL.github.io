"""Constructive approach using number-theoretic interval selection to maximize set size."""
import time
import random

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 50.0  # safety margin before 60s wall-clock kill

def _constructive_set(k, n_target):
    """
    Construct a set of n_target distinct integers from interval [2^k, 2^k + 2^{k-1})
    such that subset sums of different cardinalities fall in disjoint ranges,
    avoiding divisibility issues.
    
    For elements in [2^k, 2^k + 2^{k-1}):
    - Sum of m elements is in [m*2^k, m*(2^k + 2^{k-1}))
    - For different m, these ranges are disjoint if m*2^k >= (m-1)*(2^k + 2^{k-1})
      i.e., m*2^k >= m*2^k + m*2^{k-1} - 2^k - 2^{k-1}
      i.e., 0 >= m*2^{k-1} - 2^k - 2^{k-1}
      i.e., 2^k + 2^{k-1} >= m*2^{k-1}
      i.e., 2^{k+1} + 2^k >= m*2^{k-1}
      i.e., 4*2^{k-1} + 2*2^{k-1} >= m*2^{k-1}
      i.e., 6 >= m
      
    So for m <= 6, the ranges for different cardinalities are disjoint!
    This means no sum of m elements can divide a sum of m' elements (m != m')
    because they fall in disjoint intervals.
    
    We still need to check divisibility within the same cardinality,
    but we can use random selection and verify.
    """
    low = 2 ** k
    high = low + (1 << (k - 1)) - 1  # inclusive upper bound
    
    # Generate candidate pool from the interval
    candidates = list(range(low, min(high + 1, 257)))
    if not candidates:
        return []
    
    # Shuffle for randomness
    random.shuffle(candidates)
    
    # Greedily select elements
    selected = []
    sums_by_cardinality = {}  # cardinality -> set of sums
    
    for cand in candidates:
        if len(selected) >= n_target:
            break
        if time.time() > DEADLINE_SECONDS - 10:
            break
        
        # Try to add cand
        valid = True
        new_sums_by_card = {m: set() for m in sums_by_cardinality}
        
        # Generate new sums: for each existing sum of cardinality m, add cand to get sum of cardinality m+1
        for m, sum_set in sums_by_cardinality.items():
            new_sums_by_card[m] = set(sum_set)  # copy
            for s in sum_set:
                if m + 1 not in new_sums_by_card:
                    new_sums_by_card[m + 1] = set()
                new_sums_by_card[m + 1].add(s + cand)
        
        # Also add cand itself (cardinality 1)
        if 1 not in new_sums_by_card:
            new_sums_by_card[1] = set()
        new_sums_by_card[1].add(cand)
        
        # Check for duplicate sums across all cardinalities (invalidity condition)
        all_sums = set()
        for m, sum_set in new_sums_by_card.items():
            if len(all_sums.union(sum_set)) != len(all_sums) + len(sum_set):
                # Duplicate sum found
                valid = False
                break
            all_sums.update(sum_set)
        
        if valid:
            # Check divisibility between different cardinalities
            # Optimization: Only check if ranges might interact
            # For now, check all pairs of sums from different cardinalities
            # This is still O(N^2) but N is smaller if we prune early?
            # Actually, let's just check all pairs in all_sums for divisibility
            # But we know within same cardinality is safe.
            # Faster divisibility check
            sums_set = all_sums
            max_sum = max(sums_set) if sums_set else 0
            # To speed up, we can just check if any sum is a multiple of another
            # Iterate through sums, and for each, check if any multiple exists in the set
            # Since sums are positive, s < t implies t = k*s for k >= 2
            for s in sums_set:
                if s == 0: continue
                multiple = 2 * s
                while multiple <= max_sum:
                    if multiple in sums_set:
                        valid = False
                        break
                    multiple += s
                if not valid:
                    break
        
        if valid:
            selected.append(cand)
            sums_by_cardinality = new_sums_by_card
    
    return selected

def _verify_set(elements):
    """Verify that a set satisfies the divisibility constraint."""
    n = len(elements)
    if n == 0:
        return True
    
    # Generate all subset sums by cardinality
    sums_by_card = {0: {0}}
    for e in elements:
        new_sums = {}
        for m, sset in sums_by_card.items():
            if m not in new_sums:
                new_sums[m] = set()
            new_sums[m].update(sset)
            for s in sset:
                if m + 1 not in new_sums:
                    new_sums[m + 1] = set()
                new_sums[m + 1].add(s + e)
        sums_by_card = new_sums
    
    # Check for duplicate sums across all subsets (not just same cardinality)
    all_sums = set()
    for m, sset in sums_by_card.items():
        if m > 0:
            all_sums.update(sset)
    
    # If we have fewer unique sums than total subsets (2^n - 1), there are duplicates
    # Total non-empty subsets is 2^n - 1
    if len(all_sums) != (2 ** n) - 1:
        return False
        
    # Check divisibility
    sums_set = all_sums
    max_sum = max(sums_set) if sums_set else 0
    for s in sums_set:
        if s == 0: continue
        multiple = 2 * s
        while multiple <= max_sum:
            if multiple in sums_set:
                return False
            multiple += s
            
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    # Try different values of k and n_target
    # k determines the interval [2^k, 2^k + 2^{k-1})
    # We want elements in 1..256, so 2^k + 2^{k-1} - 1 <= 256
    # For k=7: [128, 191] - 64 elements available
    # For k=8: [256, 383] - only 256 fits
    # For k=6: [64, 95] - 32 elements available
    # For k=5: [32, 47] - 16 elements available
    
    # Try to maximize n_target
    for k in [7, 6, 5, 4, 3, 2, 1]:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Try different target sizes
        # Increase targets to maximize size, relying on faster divisibility check
        for n_target in [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            
            # Run multiple attempts with different random seeds
            # More attempts for larger targets to improve success rate
            attempts = 10 if n_target >= 12 else 5
            for attempt in range(attempts):
                if time.time() - start_time > DEADLINE_SECONDS - 10:
                    break
                
                candidate_set = _constructive_set(k, n_target)
                
                if len(candidate_set) > len(best_set):
                    # Verify the set
                    if _verify_set(candidate_set):
                        best_set = candidate_set.copy()
    
    # Fallback: try a simple search if constructive approach didn't find much
    if len(best_set) < 5:
        # Simple greedy with random restarts
        for restart in range(20):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            
            current = []
            current_sums = {0}
            
            for cand in candidates:
                if time.time() - start_time > DEADLINE_SECONDS - 5:
                    break
                
                # Check if cand can be added
                new_sums = set(current_sums)
                valid = True
                for s in current_sums:
                    ns = s + cand
                    if ns in current_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                
                if not valid:
                    continue
                
                # Check divisibility
                sums_set = {s for s in new_sums if s > 0}
                max_sum = max(sums_set) if sums_set else 0
                for s in sums_set:
                    if s == 0: continue
                    multiple = 2 * s
                    while multiple <= max_sum:
                        if multiple in sums_set:
                            valid = False
                            break
                        multiple += s
                    if not valid:
                        break
                
                if valid:
                    current.append(cand)
                    current_sums = new_sums
            
            if len(current) > len(best_set):
                if _verify_set(current):
                    best_set = current.copy()
    
    # Ensure we return a valid non-empty set
    if not best_set:
        best_set = [31, 32, 34]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
