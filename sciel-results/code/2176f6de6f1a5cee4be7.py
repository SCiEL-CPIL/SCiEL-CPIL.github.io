from fractions import Fraction
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s kill

    N = 24
    elements = list(range(1, N + 1))
    recip = {x: Fraction(1, x) for x in elements}

    def find_pattern_masks(target, universe):
        """All non-empty subsets of universe summing to target, as global bitmasks."""
        n = len(universe)
        mid = n // 2
        left = universe[:mid]
        right = universe[mid:]

        left_sums = {Fraction(0, 1): [0]}
        for i, elem in enumerate(left):
            r = recip[elem]
            for s, masks in list(left_sums.items()):
                ns = s + r
                if ns not in left_sums:
                    left_sums[ns] = []
                for m in masks:
                    left_sums[ns].append(m | (1 << i))

        right_sums = {Fraction(0, 1): [0]}
        for i, elem in enumerate(right):
            r = recip[elem]
            for s, masks in list(right_sums.items()):
                ns = s + r
                if ns not in right_sums:
                    right_sums[ns] = []
                for m in masks:
                    right_sums[ns].append(m | (1 << i))

        results = set()
        for ls, lmasks in left_sums.items():
            needed = target - ls
            if needed in right_sums:
                for lm in lmasks:
                    for rm in right_sums[needed]:
                        if lm == 0 and rm == 0:
                            continue
                        gmask = lm
                        for j, ch in enumerate(bin(rm)[2:][::-1]):
                            if ch == '1':
                                gmask |= (1 << (mid + j))
                        results.add(gmask)
        return list(results)

    # Precompute forbidden patterns (conflict hypergraph edges)
    patterns_of = {}
    for x in elements:
        others = [e for e in elements if e != x]
        patterns_of[x] = find_pattern_masks(recip[x], others)

    # patterns_containing[e]: (x, pm) where e in pm and x != e (adding e may complete pm for x)
    patterns_containing = {e: [] for e in elements}
    for x in elements:
        for pm in patterns_of[x]:
            for e in elements:
                if e != x and (pm & (1 << (e - 1))):
                    patterns_containing[e].append((x, pm))

    def can_add(mask, e):
        bit = 1 << (e - 1)
        if mask & bit:
            return True
        new_mask = mask | bit
        for pm in patterns_of[e]:
            if (new_mask & pm) == pm:
                return False
        for (x, pm) in patterns_containing[e]:
            if (mask & (1 << (x - 1))) and (new_mask & pm) == pm:
                return False
        return True

    def is_valid_mask(mask):
        for x in elements:
            if not (mask & (1 << (x - 1))):
                continue
            for pm in patterns_of[x]:
                if (mask & pm) == pm:
                    return False
        return True

    def popcount(m):
        return bin(m).count('1')

    def greedy(order):
        mask = 0
        for e in order:
            if time.time() > deadline:
                break
            if can_add(mask, e):
                mask |= (1 << (e - 1))
        return mask

    best_mask = 0
    best_count = 0

    def consider(mask):
        nonlocal best_mask, best_count
        if popcount(mask) > best_count:
            best_count = popcount(mask)
            best_mask = mask

    # Deterministic orderings
    degree = {e: len(patterns_of[e]) + len(patterns_containing[e]) for e in elements}
    orderings = [
        sorted(elements, reverse=True),
        sorted(elements),
        sorted(elements, key=lambda e: -degree[e]),
        sorted(elements, key=lambda e: degree[e]),
        sorted(elements, key=lambda e: e % 7),
    ]
    for o in orderings:
        consider(greedy(o))

    # Randomized restarts
    rng = random.Random(12345)
    restarts = 3000
    for _ in range(restarts):
        if time.time() > deadline - 25:
            break
        o = elements[:]
        rng.shuffle(o)
        consider(greedy(o))

    # Local search: add any excluded element that fits
    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        for e in sorted(elements, reverse=True):
            if not (best_mask & (1 << (e - 1))):
                if can_add(best_mask, e):
                    best_mask |= (1 << (e - 1))
                    best_count = popcount(best_mask)
                    improved = True

    # Branch-and-bound DFS polish (least-constrained first, include-first)
    dfs_order = sorted(elements, key=lambda e: degree[e])
    nodes = [0]
    node_cap = 4_000_000
    sys.setrecursionlimit(100000)

    def dfs(idx, mask, count):
        if time.time() > deadline - 5 or nodes[0] > node_cap:
            return
        nodes[0] += 1
        if idx == len(dfs_order):
            consider(mask)
            return
        if count + (len(dfs_order) - idx) <= best_count:
            return
        e = dfs_order[idx]
        bit = 1 << (e - 1)
        if can_add(mask, e):
            dfs(idx + 1, mask | bit, count + 1)
        dfs(idx + 1, mask, count)

    if time.time() < deadline - 25:
        dfs(0, 0, 0)

    # Final local-search pass
    for e in sorted(elements, reverse=True):
        if not (best_mask & (1 << (e - 1))):
            if can_add(best_mask, e):
                best_mask |= (1 << (e - 1))
    best_count = popcount(best_mask)

    result = [e for e in elements if best_mask & (1 << (e - 1))]
    if not result:
        result = [24]
    return sorted(result)
