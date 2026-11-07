"""Rewritten program: cardinality-separated subset sums via greedy in decreasing order."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s wall-clock kill

    # We want distinct integers in 1..256.
    # Strategy: try to build a set S of distinct ints from {1..256} such that
    # for any two subsets A, B with |A| != |B|, sum(A) != sum(B).
    # Equivalently: the set of subset sums of cardinality k (for k=0..n) are all
    # disjoint across different k values.
    #
    # Key insight: if we pick numbers that are "superincreasing" enough, the subset sums
    # of different cardinalities won't collide. A greedy approach from largest to smallest
    # tends to work well because larger numbers create larger gaps.
    #
    # We'll use a DP-based check: for each cardinality k, compute the set of achievable sums.
    # Then verify that these sets are pairwise disjoint.
    #
    # To maximize size, we try adding elements greedily in decreasing order, checking
    # at each step whether the cardinality-separation constraint is maintained.

    # Candidate pool: 256 down to 1
    candidates = list(range(256, 0, -1))

    # State: for each cardinality k (0..current_max), the set of achievable sums
    # We maintain dp[k] = set of sums achievable with exactly k elements from the chosen set
    # dp[0] = {0}

    chosen = []
    # dp[k] is a set of ints; dp[0] = {0}
    dp = [set() for _ in range(61)]  # up to 60 elements max
    dp[0].add(0)
    max_k = 0

    for cand in candidates:
        if time.time() > deadline:
            break

        # Try adding cand: new_dp[k] = dp[k] ∪ {s + cand for s in dp[k-1]}
        # We need to check that for all k, new_dp[k] doesn't intersect with new_dp[j] for j != k
        # But actually, the constraint is: for any two subsets with different cardinalities,
        # their sums differ. This means: for all k != j, new_dp[k] ∩ new_dp[j] = ∅.
        
        # Compute new dp
        new_dp = [set(s) for s in dp]  # copy existing
        # Update in reverse order to avoid using cand multiple times
        for k in range(max_k + 1, 0, -1):
            for s in dp[k-1]:
                new_dp[k].add(s + cand)
        
        new_max_k = max_k + 1

        # Check disjointness: for all k != j, new_dp[k] ∩ new_dp[j] = ∅
        valid = True
        for k in range(1, new_max_k + 1):
            if not new_dp[k]:
                continue
            for j in range(0, k):
                if new_dp[j] & new_dp[k]:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            chosen.append(cand)
            dp = new_dp
            max_k = new_max_k

    # Ensure we return a list of Python ints
    result = [int(x) for x in chosen]
    return result
# EVOLVE-BLOCK-END
