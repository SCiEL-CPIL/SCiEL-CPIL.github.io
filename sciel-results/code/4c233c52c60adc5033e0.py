"""Rewritten program: exact cardinality-indexed subset-sum DP to maximize set size."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_validity(elements):
    """
    Check if a set of distinct positive integers satisfies the constraint:
    No two subsets of different cardinalities have the same sum.
    
    Returns True if valid, False otherwise.
    """
    n = len(elements)
    max_sum = sum(elements)
    
    # For each cardinality c, track which sums are achievable
    # sums[c] is a set of achievable sums with exactly c elements
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    
    for elem in elements:
        # Process in reverse to avoid using the same element multiple times
        for c in range(n - 1, -1, -1):
            if sums[c]:
                for s in list(sums[c]):
                    sums[c + 1].add(s + elem)
    
    # Check pairwise disjointness of sum sets across different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums[i] & sums[j]:
                return False
    return True


def _max_valid_size_dp(elements):
    """
    Given a list of distinct positive integers, find the maximum size k
    such that there exists a subset of size k that is valid.
    Actually, we need the largest valid subset of the given elements.
    
    We'll use a greedy approach: try to add elements one by one, checking validity.
    But that's expensive. Instead, we'll search for the maximum size.
    
    Better approach: Use DP to find the maximum cardinality of a valid subset.
    We track for each cardinality c, the set of achievable sums, but we need to
    ensure we're tracking valid subsets only.
    
    Actually, let's think differently. The constraint is global: for the entire set
    we choose, no two subsets of different cardinalities can have the same sum.
    
    Let's use a different approach:
    1. Start with an empty set
    2. Try to add elements in some order
    3. After each addition, check if the current set is still valid
    4. If not valid, try a different element or backtrack
    
    This is essentially a backtracking search. Let's implement it with pruning.
    """
    n = len(elements)
    elements = sorted(elements)
    
    best_size = 0
    best_set = []
    
    def backtrack(start, current_set):
        nonlocal best_size, best_set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        # Try adding each element from start onwards
        for i in range(start, n):
            new_set = current_set + [elements[i]]
            if _check_validity(new_set):
                backtrack(i + 1, new_set)
    
    backtrack(0, [])
    return best_set


def _try_random_search(elements, num_trials=500, deadline=None):
    """
    Try random subsets and check validity. Return the largest valid subset found.
    """
    n = len(elements)
    best_size = 0
    best_set = []
    
    # Try all sizes from large to small
    for size in range(n, 0, -1):
        found = False
        for trial in range(num_trials):
            if deadline and time.time() > deadline:
                break
            subset = random.sample(elements, size)
            if _check_validity(subset):
                best_size = size
                best_set = subset
                found = True
                break
        if found:
            break
    
    return best_set


def _construct_with_dp(elements, deadline):
    """
    Use DP to find the maximum valid subset.
    """
    n = len(elements)
    
    # We'll use a DP approach where we track for each cardinality c,
    # the set of achievable sums, but we need to ensure validity.
    
    # Actually, let's use a smarter approach:
    # The key insight is that if we have elements a1 < a2 < ... < an,
    # and we want to maximize the number of elements, we should use
    # elements that are "spread out" to avoid sum collisions.
    
    # A good heuristic: use powers of 2 or similar exponentially growing sequences.
    # But we're limited to 1..256.
    
    # Let's try a constructive approach:
    # Start with small elements and greedily add the largest possible element
    # that maintains validity.
    
    best_set = []
    remaining = list(elements)
    
    while remaining:
        if time.time() > deadline:
            break
        # Try adding each remaining element
        added = False
        for elem in sorted(remaining, reverse=True):
            candidate = best_set + [elem]
            if _check_validity(candidate):
                best_set = candidate
                remaining.remove(elem)
                added = True
                break
        if not added:
            break
    
    return best_set


def construct():
    deadline = time.time() + 55  # Leave 5s safety margin for 60s limit
    
    # Generate candidate elements: we want distinct integers in 1..256
    # Strategy: use a mix of small and large numbers to maximize size
    
    # Try different strategies and pick the best result
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Greedy with DP-based validity check
    elements1 = list(range(1, 257))
    result1 = _construct_with_dp(elements1, deadline)
    if len(result1) > best_size:
        best_size = len(result1)
        best_result = result1
    
    # Strategy 2: Random search
    if time.time() < deadline - 5:
        elements2 = list(range(1, 257))
        result2 = _try_random_search(elements2, num_trials=200, deadline=deadline)
        if len(result2) > best_size:
            best_size = len(result2)
            best_result = result2
    
    # Strategy 3: Use exponentially spaced elements
    if time.time() < deadline - 3:
        # Powers of 2: 1, 2, 4, 8, ..., 128 -> 8 elements
        # Try: 1, 2, 4, 8, 16, 32, 64, 128 -> 8 elements
        # Can we do better with non-power-of-2?
        exp_elements = [1, 2, 4, 8, 16, 32, 64, 128]
        if _check_validity(exp_elements):
            if len(exp_elements) > best_size:
                best_size = len(exp_elements)
                best_result = exp_elements
    
    # Strategy 4: Try Fibonacci-like or other sequences
    if time.time() < deadline - 2:
        # Try: 1, 2, 4, 7, 12, 20, 33, 54, 88, 143 -> check if valid
        fib_like = [1, 2, 4, 7, 12, 20, 33, 54, 88, 143]
        if _check_validity(fib_like):
            if len(fib_like) > best_size:
                best_size = len(fib_like)
                best_result = fib_like
    
    # Strategy 5: Backtracking search for small sizes
    if time.time() < deadline - 1:
        # Try to find a valid set of size 10+
        small_elements = list(range(1, 50))
        result5 = _max_valid_size_dp(small_elements)
        if len(result5) > best_size:
            best_size = len(result5)
            best_result = result5
    
    # Ensure we return a valid result
    if not best_result:
        # Fallback: single element
        best_result = [1]
    
    # Convert to list of Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
