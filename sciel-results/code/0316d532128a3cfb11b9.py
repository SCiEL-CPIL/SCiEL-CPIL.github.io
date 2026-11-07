"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
from math import gcd

DEADLINE_SECONDS = 85.0

def _divides(a, b):
    """True if a divides b (a,b positive ints, a <= b)."""
    if a == b:
        return False
    return b % a == 0

def _has_divisibility_conflict(sums):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    Uses sorted array and, for each a, checks multiples via divisibility
    test on candidates. For efficiency we check each a against each b>a
    but break early; the sum-set size is bounded by 2^n but in practice
    for n<=15 it's <= 32767, and we early-exit on first conflict.
    """
    arr = sorted(sums)
    n = len(arr)
    if n < 2:
        return False
    # For each a, check if any b > a is divisible by a.
    # To speed up, for each a we only need to check b in arr that are >= 2a.
    # Use a set for O(1) membership of multiples? But arr may not contain all multiples.
    # Instead: for each a, iterate b from index of first element >= 2a.
    # This is O(n^2) worst case but with early exit and small n it's fine.
    for i in range(n):
        a = arr[i]
        # b must be >= 2a and b % a == 0
        # Find start index: first element >= 2a
        lo = i + 1
        # simple scan
        for j in range(lo, n):
            b = arr[j]
            if b >= 2 * a and b % a == 0:
                return True
        # If 2a > arr[-1], no b can work for this a
        if 2 * a > arr[-1]:
            break
    return False

def _can_add(current_sums, x):
    """
    Check if adding element x to a set with nonempty-subset-sums `current_sums`
    keeps the divisibility-incomparability property.
    New sums = current_sums ∪ {x} ∪ {s + x for s in current_sums}.
    """
    new_sums = set(current_sums)
    new_sums.add(x)
    for s in current_sums:
        new_sums.add(s + x)
    if _has_divisibility_conflict(new_sums):
        return None
    return new_sums

def _greedy_construct(candidates, start_time, deadline):
    """
    Greedy: try to add candidates in given order, keep the set that grows largest.
    Returns (set_of_elements, set_of_sums).
    """
    elements = []
    sums = set()
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums = _can_add(sums, x)
        if new_sums is not None:
            elements.append(x)
            sums = new_sums
    return elements, sums

def _random_order(pool):
    """Return a random permutation of pool."""
    lst = list(pool)
    random.shuffle(lst)
    return lst

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate pool: integers in 1..256.
    # To maximize size, we want many elements. Larger elements reduce subset-sum
    # collisions and divisibility conflicts? Actually smaller elements create more
    # subset sums (more chances for divisibility). Larger elements create sums that
    # are large and less likely to divide each other. But we need distinct ints in 1..256.
    # Strategy: try multiple random orderings of a subset of candidates.
    # Use a pool of candidates. We'll use all 1..256 but random orderings.

    pool = list(range(1, 257))

    best_elements = []
    best_sums = set()

    # First: a deterministic greedy with descending order (larger elements first)
    # Larger elements first may allow more elements because sums grow faster and
    # divisibility conflicts are less likely among large numbers.
    order = sorted(pool, reverse=True)
    elems, sums = _greedy_construct(order, start_time, deadline)
    if len(elems) > len(best_elements):
        best_elements = elems
        best_sums = sums

    # Also try ascending order
    if time.time() < deadline:
        order = sorted(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums

    # Random restarts
    max_restarts = 2000
    for r in range(max_restarts):
        if time.time() > deadline - 0.5:
            break
        order = _random_order(pool)
        elems, sums = _greedy_construct(order, start_time, deadline)
        if len(elems) > len(best_elements):
            best_elements = elems
            best_sums = sums
            # If we hit a very large size, we can stop early? No, keep searching.

    # Final: try local improvement on best set by adding remaining elements in random order
    if time.time() < deadline - 0.5:
        used = set(best_elements)
        remaining = [x for x in pool if x not in used]
        random.shuffle(remaining)
        elems = list(best_elements)
        sums = set(best_sums)
        for x in remaining:
            if time.time() > deadline - 0.2:
                break
            new_sums = _can_add(sums, x)
            if new_sums is not None:
                elems.append(x)
                sums = new_sums
        if len(elems) > len(best_elements):
            best_elements = elems

    # Ensure return type is list of Python ints
    return list(best_elements)
