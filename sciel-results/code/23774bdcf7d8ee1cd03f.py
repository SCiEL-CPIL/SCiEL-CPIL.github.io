"""Rewritten program: bitset-DP accelerated greedy + randomized trials with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

# Hard wall-clock limit (seconds). The evaluator kills at 90s; use a safety margin.
_DEADLINE_SECONDS = 85.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE_SECONDS - _elapsed()

# ---------------------------------------------------------------------------
# Fast violation check using bitset DP over the "others" set.
# For a candidate set S and a target element a in S, we ask:
#   does there exist a subset T of S\{a} with |T|>=2 and sum(T) == |T|*a ?
# Equivalently, with b_i = x_i - a for x_i in S\{a}, we need a subset T with
#   sum_{i in T} b_i == 0 and |T| >= 2.
#
# We do DP over (sum_offset, count) where sum_offset is the shifted sum.
# Since x_i in [1,512] and a in [1,512], b_i in [-511, 511].
# For |others| = m, the min possible sum is -511*m and max is 511*m.
# But we only care about sum==0. We can use a dict or a bounded array.
#
# Better approach: use a set of reachable (sum, count) pairs, but prune.
# Actually, for small m (<=47), we can use a 2D boolean array indexed by
# (sum_index, count) where sum_index maps sum value to an index.
#
# Even faster: since we only need sum==0 at the end, and counts 2..m,
# we can track for each count k the set of achievable sums.
# But sets of ints can be large. Let's use bitsets for the sum dimension.
#
# Alternative: use a list of sets, dp[k] = set of sums achievable with k elements.
# For m up to ~47, and sums in range [-511*m, 511*m], this is manageable
# if we use Python sets of ints. The number of distinct sums is at most
# (1023*m+1) which for m=47 is ~48000. That's fine.
#
# But we can do even better: note that we only need to know if 0 is achievable
# with count >= 2. So we can early-exit.
# ---------------------------------------------------------------------------

def _has_zero_sum_subset_others(others, a):
    """
    Check if there exists a subset of 'others' of size >= 2 with sum == len(subset)*a.
    Equiv: subset of 'others' with sum of (x - a) == 0 and size >= 2.
    """
    m = len(others)
    if m < 2:
        return False
    # dp[k] = set of achievable sums of (x_i - a) for subsets of size k
    # We only need k from 2 to m.
    # Use list of sets. dp[0] = {0}
    dp = [set() for _ in range(m + 1)]
    dp[0].add(0)
    for x in others:
        b = x - a
        # Process in reverse to avoid using same element twice
        for k in range(m, 0, -1):
            if not dp[k-1]:
                continue
            # Add b to each sum in dp[k-1]
            new_sums = set()
            for s in dp[k-1]:
                new_sums.add(s + b)
            if new_sums:
                dp[k].update(new_sums)
    # Check if 0 is in dp[k] for any k >= 2
    for k in range(2, m + 1):
        if 0 in dp[k]:
            return True
    return False

def _is_valid_set(S):
    """Check if no element in S is the mean of any subset of >=2 other elements."""
    for a in S:
        others = [x for x in S if x != a]
        if _has_zero_sum_subset_others(others, a):
            return False
    return True

def _greedy_build(order, max_size=48):
    """
    Greedily build a valid set by considering elements in the given 'order'.
    Returns the valid set.
    """
    selected = []
    for x in order:
        if len(selected) >= max_size:
            break
        candidate = selected + [x]
        # Check validity: for each a in candidate, check if a is mean of subset of others
        # We only need to check the new element x and existing elements (but existing were valid before)
        # Actually, adding x can invalidate existing elements too, because x can be part of a subset
        # whose mean is an existing element. So we must check all elements.
        # But for efficiency, we can check:
        # 1. Is x the mean of any subset of selected? (x = mean of T, T subset of selected, |T|>=2)
        #    Equiv: sum(T) = |T|*x, i.e., sum_{t in T}(t - x) = 0, |T|>=2
        # 2. For each a in selected, is a the mean of a subset of (selected\{a} union {x})?
        #    This is harder. But note: if a was not the mean of any subset of selected\{a} before,
        #    it could now be if x is included. So we need to check.
        #
        # For simplicity and correctness, check all elements in candidate.
        valid = True
        for a in candidate:
            others = [y for y in candidate if y != a]
            if _has_zero_sum_subset_others(others, a):
                valid = False
                break
        if valid:
            selected.append(x)
    return selected

def construct():
    """
    Returns up to 48 distinct positive integers in [1, 512] such that no element
    is the exact arithmetic mean of any subset of at least two other elements.
    Maximizes cardinality.
    """
    best_set = []
    deadline = _DEADLINE_SECONDS - 2.0  # safety margin

    # First, try a deterministic greedy in increasing order as a baseline
    order_increasing = list(range(1, 513))
    s = _greedy_build(order_increasing)
    if len(s) > len(best_set):
        best_set = s

    # Try decreasing order
    order_decreasing = list(range(512, 0, -1))
    if _time_left() > 5.0:
        s = _greedy_build(order_decreasing)
        if len(s) > len(best_set):
            best_set = s

    # Now try randomized orders. Use multiple random seeds.
    # We can also try "sparse" orderings: start with large gaps.
    rng = random.Random(12345)

    # Strategy: try many random permutations, but limit the number of trials
    # based on time budget.
    num_trials = 0
    while _time_left() > 3.0:
        # Generate a random order
        order = list(range(1, 513))
        rng.shuffle(order)
        s = _greedy_build(order)
        num_trials += 1
        if len(s) > len(best_set):
            best_set = s
            # If we've reached 48, we can stop early
            if len(best_set) >= 48:
                break

    # Final: try a few more targeted strategies if time permits
    if _time_left() > 2.0 and len(best_set) < 48:
        # Try starting with multiples of a large number to create gaps
        for base in [100, 128, 150, 200, 256]:
            if _time_left() < 1.5:
                break
            # Build an order that starts with multiples of base, then fills in
            order = []
            multiples = [x for x in range(1, 513) if x % base == 0]
            rng.shuffle(multiples)
            order.extend(multiples)
            rest = [x for x in range(1, 513) if x % base != 0]
            rng.shuffle(rest)
            order.extend(rest)
            s = _greedy_build(order)
            if len(s) > len(best_set):
                best_set = s

    return best_set

# EVOLVE-BLOCK-END
