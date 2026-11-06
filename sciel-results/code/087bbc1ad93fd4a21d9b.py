"""Optimized solver: find a nonempty subset of [1,100] with no subset summing to 200, maximizing cardinality."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s/90s kill

    # Greedy baseline (same as seed, but we will try to improve)
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    best_size = len(selected)
    best_set = set(selected)

    if time.time() > deadline - 5:
        return sorted(best_set)

    # Try to improve via local search / greedy with different orderings
    # The key insight: we want to maximize the number of elements from [1,100]
    # such that no subset sums to exactly 200.

    # A well-known approach: if we can partition [1,100] into two sets where one
    # set has all subset sums < 200 or > 200 (avoiding 200), that's ideal.
    # Actually, the complement approach: find the minimum number of elements to
    # REMOVE from [1,100] so that the remaining set has no subset summing to 200.
    # Maximize cardinality = minimize removals.

    # Let's use a different strategy: try all orderings of a few heuristics
    # and take the best.

    def greedy(order):
        """Greedy: add elements in given order if they don't create sum 200."""
        sel = []
        reach = {0}
        for x in order:
            new = {s + x for s in reach if s + x <= 200}
            if 200 not in new:
                sel.append(x)
                reach |= new
        return sel

    # Try several orderings
    orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
        # Randomized trials
    ]

    import random
    rng = random.Random(42)
    for _ in range(200):
        o = list(range(1, 101))
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() > deadline - 5:
            break
        sel = greedy(o)
        if len(sel) > best_size:
            best_size = len(sel)
            best_set = set(sel)

    # Try a smarter approach: use the fact that if we take all numbers > 100... 
    # but we're limited to [1,100]. 
    # Key observation: if we take all numbers in [1,100], the total sum is 5050, 
    # and 200 is easily achievable. We need to remove elements to break all 200-sums.

    # Alternative: try to construct a set where all subset sums are either < 200 or > 200
    # by ensuring a "gap" around 200. This is hard with dense sets.

    # Let's try a local improvement: start from best_set, try swapping elements
    # to increase size, or try adding elements that were excluded.

    # Actually, let's try a completely different construction:
    # Take all numbers in [1, 99] except those that participate in 200-sums with others.
    # This is complex. Let's try: take numbers in two ranges that can't combine to 200.
    # E.g., [1, 50] and [150, 200]... but we only have up to 100.
    # [1, 100] ∩ [150, 200] = empty. So that doesn't work directly.
    
    # What about [1, 99] minus some small set? 
    # Or: take all of [1, 100] and remove a hitting set for all 200-sum subsets.

    # Let's try: the maximum subset with no 200-sum. 
    # A known result: for [1..n] with target T, the max subset size is n - min_hitting_set.
    # For T=200, n=100.

    # Let's try a bipartite/graph approach: create a graph where vertices are elements,
    # and edges connect elements that "conflict" (i.e., there exists a subset containing
    # both that sums to 200). Then find a maximum independent set. But this is NP-hard
    # in general, and the graph is dense.

    # Given time constraints, let's try a few more heuristic orderings and a local search.

    # Local search: try to add back excluded elements one by one
    excluded = set(range(1, 101)) - best_set
    for x in list(excluded):
        if time.time() > deadline - 5:
            break
        candidate = best_set | {x}
        # Check if candidate has a subset summing to 200
        # Use DP to check
        dp = {0}
        valid = True
        for y in sorted(candidate):
            new_dp = set(dp)
            for s in dp:
                ns = s + y
                if ns == 200:
                    valid = False
                    break
                if ns <= 200:
                    new_dp.add(ns)
            if not valid:
                break
            dp = new_dp
        if valid:
            best_set = candidate
            best_size += 1

    if time.time() > deadline - 5:
        return sorted(best_set)

    # Try removing one element and adding two others (swap)
    for x in list(best_set):
        if time.time() > deadline - 5:
            break
        for y in list(excluded):
            if time.time() > deadline - 5:
                break
            for z in list(excluded):
                if y == z:
                    continue
                if time.time() > deadline - 5:
                    break
                candidate = (best_set - {x}) | {y, z}
                dp = {0}
                valid = True
                for w in sorted(candidate):
                    new_dp = set(dp)
                    for s in dp:
                        ns = s + w
                        if ns == 200:
                            valid = False
                            break
                        if ns <= 200:
                            new_dp.add(ns)
                    if not valid:
                        break
                    dp = new_dp
                if valid:
                    best_set = candidate
                    best_size += 1
                    excluded = set(range(1, 101)) - best_set
                    break

    return sorted(best_set)
# EVOLVE-BLOCK-END
