"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 256, 'maximum_size': 60}

def construct():
    # Start with a deadline guard
    start_time = time.time()
    deadline = 50.0  # Leave 10s margin
    
    # Candidate: Arithmetic Progression {226, ..., 256} of size 31
    # This set satisfies S_min(k) > S_max(k-1) for all k, 
    # which implies disjoint sum sets for different cardinalities.
    candidate = list(range(226, 257))
    
    # Verify the candidate using exact DP
    if is_valid(candidate):
        return candidate
    
    # If for some reason the AP fails (shouldn't happen), try to find a valid set greedily
    # But given the analysis, AP size 31 is the theoretical max for consecutive sets,
    # and likely for any set in 1..256.
    
    # Fallback: Return the largest valid set found by a simple greedy search
    # This is unlikely to be reached, but ensures robustness.
    best_set = []
    # Try to construct a set of size 31 greedily?
    # We know 31 works.
    
    # If AP failed, try size 30
    candidate_30 = list(range(227, 257))
    if is_valid(candidate_30):
        return candidate_30
        
    # Return empty or smallest valid if all fail (should not happen)
    return [1]

def is_valid(S):
    """
    Checks if subset sums of different cardinalities are disjoint.
    Uses DP to compute the set of sums for each cardinality.
    """
    m = len(S)
    if m == 0:
        return True
        
    # dp[k] is a set of sums achievable with k elements
    # Since sums can be large, we use sets.
    # Max sum is 31 * 256 = 7936. Small enough for sets.
    dp = [set() for _ in range(m + 1)]
    dp[0].add(0)
    
    for x in S:
        # Iterate backwards to avoid using the same element multiple times
        # in the same cardinality step (0/1 knapsack style)
        for k in range(m, 0, -1):
            if dp[k-1]:
                # Add x to all sums in dp[k-1] and add to dp[k]
                new_sums = {s + x for s in dp[k-1]}
                dp[k].update(new_sums)
                
    # Check for intersections between different cardinalities
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            if dp[i] & dp[j]:
                return False
    return True

# EVOLVE-BLOCK-END
