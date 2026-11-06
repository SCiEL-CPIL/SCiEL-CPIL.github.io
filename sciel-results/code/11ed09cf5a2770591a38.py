"""
Large reciprocal-subsum-free sets in [1,24].

Goal: return a nonempty subset S of {1,...,24} maximizing |S| such that
for every x in S, 1/x != sum_{y in T} 1/y for any T subset of S\{x}.

Strategy:
  - Use Fraction for exact arithmetic.
  - Validate a set by checking, for each x in S, whether 1/x can be
    expressed as a sum of reciprocals of distinct elements of S\{x}.
    Use a meet-in-the-middle subset-sum test over the reciprocals.
  - Search:
      1. Greedy construction starting from large numbers (superincreasing-ish).
      2. Greedy construction starting from small numbers.
      3. Greedy construction starting from medium numbers.
      4. Local search: try to add any missing element to the best set.
      5. Random restarts with random permutations.
  - Keep the best valid set found; return its sorted list.
  - Deadline guard: stop after 75 s wall-clock and return best found.
"""
import time
import random
from fractions import Fraction
from itertools import combinations

_START = None
DEADLINE = 75.0


def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE


def _subset_sum_possible(target, values):
    """
    Return True if target can be expressed as a sum of a nonempty subset of values.
    Uses meet-in-the-middle for efficiency. values is a list of Fractions.
    """
    if not values:
        return False
    n = len(values)
    if n == 0:
        return False
    if n <= 1:
        return values[0] == target
    # For small n, brute force
    if n <= 15:
        for k in range(1, n + 1):
            for combo in combinations(values, k):
                if sum(combo) == target:
                    return True
        return False
    # Meet-in-the-middle
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left half
    left_sums = set()
    left_sums.add(Fraction(0))
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    
    # Generate all subset sums for right half
    right_sums = set()
    right_sums.add(Fraction(0))
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    
    # Check if any pair sums to target
    for ls in left_sums:
        if (target - ls) in right_sums:
            # Ensure we're not using the empty subset from both sides
            # Actually, target > 0 so at least one side must be non-empty
            return True
    return False


def is_valid(S):
    """
    Check if S is reciprocal-subsum-free.
    S is a set of integers.
    """
    if not S:
        return False
    elems = list(S)
    recip = {x: Fraction(1, x) for x in elems}
    for x in elems:
        others = [recip[y] for y in elems if y != x]
        if _subset_sum_possible(recip[x], others):
            return False
    return True


def construct():
    global _START
    _START = time.time()
    
    best_set = None
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Strategy 1: Greedy from large to small
    # Start with the largest elements, try to add smaller ones
    S = set(range(24, 12, -1))  # 13..24
    # Try adding smaller elements
    for x in range(12, 0, -1):
        if _deadline_hit():
            break
        S_test = S | {x}
        if is_valid(S_test):
            S = S_test
    update_best(S)
    
    # Strategy 2: Greedy from small to large
    S = set()
    for x in range(1, 25):
        if _deadline_hit():
            break
        S_test = S | {x}
        if is_valid(S_test):
            S = S_test
    update_best(S)
    
    # Strategy 3: Greedy from middle
    for start in [8, 10, 12, 15, 18, 20]:
        if _deadline_hit():
            break
        S = set()
        for x in range(start, 25):
            if _deadline_hit():
                break
            S_test = S | {x}
            if is_valid(S_test):
                S = S_test
        for x in range(start - 1, 0, -1):
            if _deadline_hit():
                break
            S_test = S | {x}
            if is_valid(S_test):
                S = S_test
        update_best(S)
    
    # Strategy 4: Local search - try to add any missing element to best_set
    if best_set is not None:
        current = set(best_set)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for x in range(1, 25):
                if x in current:
                    continue
                S_test = current | {x}
                if is_valid(S_test):
                    current = S_test
                    improved = True
                    update_best(current)
        update_best(current)
    
    # Strategy 5: Random restarts
    random.seed(42)
    for _ in range(200):
        if _deadline_hit():
            break
        perm = list(range(1, 25))
        random.shuffle(perm)
        S = set()
        for x in perm:
            if _deadline_hit():
                break
            S_test = S | {x}
            if is_valid(S_test):
                S = S_test
        update_best(S)
    
    # Strategy 6: Another round of local search on the global best
    if best_set is not None:
        current = set(best_set)
        improved = True
        while improved and not _deadline_hit():
            improved = False
            for x in range(1, 25):
                if x in current:
                    continue
                S_test = current | {x}
                if is_valid(S_test):
                    current = S_test
                    improved = True
                    update_best(current)
    
    if best_set is None:
        return list(range(13, 25))
    
    return sorted(best_set)


if __name__ == "__main__":
    start = time.time()
    res = construct()
    elapsed = time.time() - start
    print(f"Result: {res}, Size: {len(res)}, Time: {elapsed:.2f}s")
    if is_valid(res):
        print("Valid")
    else:
        print("INVALID!")
