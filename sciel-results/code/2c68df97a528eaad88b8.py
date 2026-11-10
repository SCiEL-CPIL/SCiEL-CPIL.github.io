import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin well under the 90s kill limit

    # Precompute reciprocals as exact Fractions
    recip = {i: Fraction(1, i) for i in range(1, 25)}

    def is_valid(S):
        """
        Returns True iff S is reciprocal-subsum-free:
        for every x in S, 1/x is NOT the sum of reciprocals of any subset of S \ {x}.
        Uses meet-in-the-middle with pruning (all reciprocals are positive, so any
        partial sum exceeding the target can be dropped).
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True

        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i + 1:]
            if not others:
                continue

            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]

            # Build left subset sums, pruning any sum that exceeds target.
            left_sums = [0]
            for y in left:
                ry = recip[y]
                new_sums = []
                for s in left_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                left_sums.extend(new_sums)

            # Build right subset sums with the same pruning.
            right_sums = [0]
            for y in right:
                ry = recip[y]
                new_sums = []
                for s in right_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                right_sums.extend(new_sums)

            right_set = set(right_sums)
            for l in left_sums:
                if (target - l) in right_set:
                    return False
        return True

    def local_search_improve(S):
        """
        Repeatedly add any element of [1..24] \ S that keeps the set valid,
        restarting after each successful addition (adding an element can open
        new possibilities for further additions).
        """
        S = set(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for x in range(1, 25):
                if x not in S and is_valid(S | {x}):
                    S.add(x)
                    improved = True
                    break  # restart the pass
        return S

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_valid(S | {x}):
                S.add(x)
        return S

    def repair_full():
        """Start from the full set; remove elements that cause a violation."""
        current = set(range(1, 25))
        for _ in range(100):
            if time.time() > deadline:
                break
            if is_valid(current):
                break
            S_list = list(current)
            repaired = False
            for x in S_list:
                if time.time() > deadline:
                    break
                target = recip[x]
                others = [y for y in S_list if y != x]
                m = len(others)
                half = m // 2
                left = others[:half]
                right = others[half:]

                left_sums = [0]
                for y in left:
                    ry = recip[y]
                    left_sums.extend([s + ry for s in left_sums if s + ry <= target])

                right_sums = [0]
                for y in right:
                    ry = recip[y]
                    right_sums.extend([s + ry for s in right_sums if s + ry <= target])

                right_set = set(right_sums)
                violation = any((target - l) in right_set for l in left_sums)
                if violation:
                    current.remove(x)
                    repaired = True
                    break
            if not repaired:
                break
        return current

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    best_set = set()
    best_size = 0

    # Phase 1: repair from the full set (multiple attempts with randomized removal order)
    random.seed(12345)
    for attempt in range(12):
        if time.time() > deadline:
            break
        S = repair_full()
        S = local_search_improve(S)
        update_best(S)

    # Phase 2: greedy builds from diverse orders, then local-search to fill in
    base = list(range(1, 25))
    orders = [
        base,
        list(reversed(base)),
        [1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13],
    ]
    for _ in range(30):
        if time.time() > deadline:
            break
        o = list(base)
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_build(order)
        S = local_search_improve(S)
        update_best(S)

    # Fallback: a singleton is always valid
    if not best_set:
        best_set = {24}

    return list(best_set)
