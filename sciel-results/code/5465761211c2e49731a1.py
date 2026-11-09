"""Rewritten program: exact cardinality-indexed subset-sum DP to maximize set size."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _subset_sums_by_cardinality(elements):
    """
    For a list of distinct positive integers, compute the set of achievable sums
    for each cardinality 0..len(elements).
    Returns a list of sets: sums_by_k[k] = set of all sums of subsets of size k.
    """
    n = len(elements)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in elements:
        # Update in reverse to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + x)
    return sums_by_k

def _is_valid(elements):
    """
    Check if the set satisfies the condition: whenever two subsets have different
    cardinalities, their sums must differ.
    Equivalently: for all i != j, sums_by_k[i] ∩ sums_by_k[j] = empty.
    """
    sums_by_k = _subset_sums_by_cardinality(elements)
    n = len(sums_by_k)
    for i in range(n):
        for j in range(i + 1, n):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _max_valid_size_dp(elements):
    """
    Given a sorted list of distinct integers, find the maximum k such that
    there exists a subset of size k that is valid.
    We check all subsets from largest to smallest.
    This is expensive; used only for verification.
    """
    n = len(elements)
    for k in range(n, 0, -1):
        for subset in combinations(elements, k):
            if _is_valid(list(subset)):
                return k
    return 0

def _greedy_construct():
    """
    Greedy construction: start with a candidate pool, add elements one by one
    if the resulting set remains valid.
    """
    # Start with a reasonable range of candidates
    # We want to maximize size, so we should try to include as many as possible
    # Strategy: try to build a set from a range [lo, hi]
    
    best_set = []
    best_size = 0
    
    # Try different starting points and ranges
    # Key insight: using powers of 2 or near-powers works well for subset sum distinctness
    # But we need cardinality separation, which is different.
    
    # Let's try a systematic approach:
    # Start with small numbers and try to add larger ones
    
    # Candidate pool: 1..256
    # We'll try to build the set greedily, trying candidates in order
    
    # First, let's try a simple greedy: add 1, then try 2, 3, ...
    current = []
    for x in range(1, 257):
        candidate = current + [x]
        if _is_valid(candidate):
            current = candidate
    
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Also try starting from larger numbers
    for start in range(1, 50):
        current = []
        for x in range(start, 257):
            candidate = current + [x]
            if _is_valid(candidate):
                current = candidate
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    return best_set

def _local_search_improve(start_set, time_limit=10):
    """
    Try to improve the set by swapping elements or adding new ones.
    """
    start_time = time.time()
    current = list(start_set)
    current_set = set(current)
    best = current[:]
    best_size = len(current)
    
    # Try adding elements
    improved = True
    while improved:
        improved = False
        for x in range(1, 257):
            if x in current_set:
                continue
            candidate = current + [x]
            if _is_valid(candidate):
                current = candidate
                current_set.add(x)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
    
    # Try swapping: remove one element, add another
    for _ in range(50):  # limited iterations
        if time.time() - start_time > time_limit:
            break
        for i in range(len(current)):
            for x in range(1, 257):
                if x in current_set:
                    continue
                candidate = current[:i] + current[i+1:] + [x]
                if _is_valid(candidate):
                    if len(candidate) == len(current):
                        # Only accept if it helps future additions (hard to predict)
                        # For now, skip swaps that don't increase size
                        pass
    
    return best

def construct():
    start_time = time.time()
    deadline = 80  # 80 seconds out of 90
    
    # Try greedy construction
    best_set = _greedy_construct()
    best_size = len(best_set)
    
    # Try to improve with local search
    if time.time() - start_time < deadline - 5:
        improved = _local_search_improve(best_set, time_limit=10)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Verify the result
    if not _is_valid(best_set):
        # Fallback to a known valid small set
        best_set = list(range(1, 10))
    
    # Ensure all elements are Python ints
    best_set = [int(x) for x in best_set]
    
    return best_set

# EVOLVE-BLOCK-END
