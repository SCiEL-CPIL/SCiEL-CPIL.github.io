"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import defaultdict

DEADLINE_MARGIN = 5.0  # seconds before the 90s hard kill

def _subset_sum_sets_by_cardinality(nums):
    """
    For a sorted list of distinct positive ints, compute for each cardinality k
    the set of all subset sums of size k.
    Returns a list S where S[k] is a set of ints.
    S[0] = {0}.
    """
    n = len(nums)
    max_k = n
    # S[k] = set of sums
    S = [set() for _ in range(max_k + 1)]
    S[0].add(0)
    
    for x in nums:
        # iterate k descending to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            if S[k]:
                # new sums for cardinality k+1
                for s in S[k]:
                    S[k + 1].add(s + x)
    return S

def _check_cardinality_separation(nums):
    """
    Check if for all k != l, S[k] and S[l] are disjoint.
    """
    S = _subset_sum_sets_by_cardinality(nums)
    for k in range(len(S)):
        for l in range(k + 1, len(S)):
            if S[k] & S[l]:
                return False
    return True

def _reverse_greedy(N, max_size, deadline):
    """
    Start with the largest integers and greedily add the next largest
    that preserves cardinality separation.
    """
    # Candidate pool: distinct integers in 1..N, sorted descending
    pool = list(range(N, 0, -1))
    result = []
    used = set()
    
    # We want to maximize size. Try to add from largest to smallest.
    # But we need to check validity.
    # A better heuristic: start with the largest possible set?
    # Let's try adding in descending order.
    
    # Actually, "Reverse greedy with n largest integers" suggests:
    # Take the largest n integers, check if valid. If not, replace one with a smaller one?
    # Or simply: iterate candidates from largest to smallest, add if valid.
    
    for x in pool:
        if time.time() > deadline:
            break
        if len(result) >= max_size:
            break
            
        candidate = result + [x]
        # Sort for consistency (though order doesn't matter for sums)
        # We can just check the set of values
        if _check_cardinality_separation(candidate):
            result.append(x)
    
    # The result is in descending order. The task doesn't require sorting, but let's return sorted ascending for cleanliness.
    return sorted(result)

def _local_search(nums, deadline):
    """
    Try to improve the size or just validate.
    Since the objective is purely size, and reverse greedy likely finds a local maximum,
    we can try to swap elements to free up space for more elements?
    Actually, if we have a set of size K, can we add one more?
    The reverse greedy already tries to add the largest available.
    If it fails to add any more, we are at a local maximum for that ordering.
    
    A different structural change: Randomized restarts.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    N = 256
    max_size = 60
    
    best_set = []
    
    # 1. Deterministic Reverse Greedy
    greedy_result = _reverse_greedy(N, max_size, deadline)
    if len(greedy_result) > len(best_set):
        best_set = greedy_result
    
    if time.time() > deadline:
        return best_set
    
    # 2. Randomized Hill Climbing / Restart
    # Try to find a larger set by random permutations or random subsets.
    # Given the constraint is strict (disjoint sum sets across cardinalities),
    # valid sets are likely rare or structured.
    
    # Let's try a different greedy: Smallest first?
    # "Largest integers" was the insight. Let's stick with that but maybe perturb.
    
    # Alternative: Start with a known valid small set and expand?
    # Or use the property that powers of 2 (1, 2, 4, 8...) have unique subset sums for ALL subsets, hence cardinality separation holds trivially.
    # Powers of 2 up to 256: 1, 2, 4, 8, 16, 32, 64, 128. Size 8.
    # The insight said "n largest integers" works better.
    # Let's check if {197..256} (60 numbers) works?
    # Sum of 1 smallest (197) vs Sum of 2 smallest (197+198=395).
    # Max sum of 1 is 256. Min sum of 2 is 395. Disjoint.
    # Max sum of 2 is 256+255=511. Min sum of 3 is 197+198+199=594. Disjoint.
    # It seems consecutive large integers might work because the gap between min sum of k+1 and max sum of k is large enough.
    # Min sum of k+1 from {M..N} is sum of k+1 smallest: (k+1)M + k(k+1)/2.
    # Max sum of k from {M..N} is sum of k largest: kN - k(k-1)/2.
    # We need MinSum(k+1) > MaxSum(k).
    # (k+1)M + k(k+1)/2 > kN - k(k-1)/2
    # For M=197, N=256, k=59 (max k for size 60 is 59, comparing S[59] and S[60]):
    # MinSum(60) = 60*197 + 60*59/2 = 11820 + 1770 = 13590.
    # MaxSum(59) = 59*256 - 59*58/2 = 15104 - 1711 = 13393.
    # 13590 > 13393. So it holds!
    # Let's check k=1 vs k=2:
    # MinSum(2) = 2*197 + 1 = 395.
    # MaxSum(1) = 256.
    # 395 > 256. Holds.
    
    # So the set of the 60 largest integers {197, ..., 256} should be valid.
    # Let's verify this explicitly.
    
    candidate = list(range(197, 257))
    if _check_cardinality_separation(candidate):
        if len(candidate) > len(best_set):
            best_set = candidate
        return best_set # We found size 60, which is the max possible.
    
    # If it failed (unlikely based on math), fall back to greedy result.
    return best_set

# EVOLVE-BLOCK-END
