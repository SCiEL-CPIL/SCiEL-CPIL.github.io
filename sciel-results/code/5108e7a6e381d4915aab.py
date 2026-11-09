"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # seconds, safety margin for 60s/90s limits


def _pair_sums_ok(lst):
    """Check if all pair sums (including a+a) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            b = lst[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a != b."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i + 1, len(lst)):
            b = lst[j]
            d.add(b - a)
            d.add(a - b)
    return d


def _fast_greedy(order, deadline):
    """
    Greedily build two Sidon sets A and B following the given element order.
    For each element, try to add it to A, then to B, then skip.
    Returns (A, B) as lists.
    """
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()

    start = time.time()

    for x in order:
        if time.time() - start > deadline:
            break

        # Try adding x to A
        can_add_A = True
        new_sums_A = set()
        for a in A:
            s = a + x
            if s in sums_A or s in new_sums_A:
                can_add_A = False
                break
            new_sums_A.add(s)
        s2 = 2 * x
        if can_add_A and (s2 in sums_A or s2 in new_sums_A):
            can_add_A = False

        if can_add_A:
            # Check difference disjointness
            new_diffs = set()
            for a in A:
                d1 = x - a
                d2 = a - x
                new_diffs.add(d1)
                new_diffs.add(d2)
            if new_diffs & diffs_B:
                can_add_A = False

        if can_add_A:
            A.append(x)
            sums_A.update(new_sums_A)
            sums_A.add(s2)
            for a in A:
                if a != x:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            continue

        # Try adding x to B
        can_add_B = True
        new_sums_B = set()
        for b in B:
            s = b + x
            if s in sums_B or s in new_sums_B:
                can_add_B = False
                break
            new_sums_B.add(s)
        s2b = 2 * x
        if can_add_B and (s2b in sums_B or s2b in new_sums_B):
            can_add_B = False

        if can_add_B:
            new_diffs = set()
            for b in B:
                d1 = x - b
                d2 = b - x
                new_diffs.add(d1)
                new_diffs.add(d2)
            if new_diffs & diffs_A:
                can_add_B = False

        if can_add_B:
            B.append(x)
            sums_B.update(new_sums_B)
            sums_B.add(s2b)
            for b in B:
                if b != x:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)

    return A, B


def construct():
    start = time.time()
    best_A = []
    best_B = []
    best_score = 0

    # Multiple random restarts with different orderings
    # Use a mix of strategies: random, sorted, reverse, random with bias
    rng = random.Random(42)

    for trial in range(200):
        if time.time() - start > DEADLINE - 5.0:
            break

        # Generate a random ordering
        if trial % 4 == 0:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif trial % 4 == 1:
            # Sorted ascending
            order = list(range(1, N + 1))
        elif trial % 4 == 2:
            # Sorted descending
            order = list(range(N, 0, -1))
        else:
            # Random with slight bias toward smaller numbers
            order = list(range(1, N + 1))
            rng.shuffle(order)
            # Partially sort to favor smaller elements
            order.sort(key=lambda x: x + rng.random() * 50)

        A, B = _fast_greedy(order, DEADLINE - (time.time() - start))

        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B

            # Early exit if we hit the maximum possible
            if best_score >= 2 * MAX_EACH:
                break

    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
