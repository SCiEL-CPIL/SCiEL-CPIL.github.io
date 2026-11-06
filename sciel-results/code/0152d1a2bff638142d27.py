import time
from math import lcm
from functools import reduce

def construct():
    start = time.time()
    deadline = start + 85.0

    L = reduce(lcm, range(1, 23))
    recip = {i: L // i for i in range(1, 23)}

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_len = len(best)

    def can_add(sums, r):
        # collision if exists s1,s2 in sums with s1 - s2 == r  (i.e., s2 + r in sums)
        for s in sums:
            if s + r in sums:
                return False
        return True

    def compute_sums(chosen_set):
        sums = {0}
        for i in chosen_set:
            r = recip[i]
            new_sums = set()
            for s in sums:
                new_sums.add(s + r)
            sums = sums | new_sums
        return sums

    def local_search(chosen):
        """Try to improve by swapping elements in/out."""
        nonlocal best, best_len
        chosen_set = set(chosen)
        sums = compute_sums(chosen_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try adding one element
            for i in range(1, 23):
                if i in chosen_set:
                    continue
                r = recip[i]
                if can_add(sums, r):
                    chosen_set.add(i)
                    sums = sums | {s + r for s in sums}
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one, add another
            for out_i in list(chosen_set):
                if time.time() > deadline - 2.0:
                    break
                # Compute sums without out_i
                r_out = recip[out_i]
                # Rebuild sums without out_i
                temp_set = chosen_set - {out_i}
                temp_sums = {0}
                for j in temp_set:
                    r_j = recip[j]
                    temp_sums = temp_sums | {s + r_j for s in temp_sums}
                for in_i in range(1, 23):
                    if in_i in temp_set:
                        continue
                    r_in = recip[in_i]
                    if can_add(temp_sums, r_in):
                        # Check that the new set is strictly larger or same size but we prefer larger
                        # Since we removed one and added one, size is same. 
                        # We only do this if it helps future additions? Hard to tell.
                        # Let's stick to pure additions first for simplicity and safety.
                        # Swapping might lead to dead ends. 
                        # For now, skip complex swaps to avoid bugs, focus on fast greedy + backtrack.
                        pass
            # If no addition found, try a simple swap to unblock? 
            # Actually, if we can't add anything, we are likely at a local optimum for this order.
            # Let's break and rely on DFS for global search.
            break
        
        if len(chosen_set) > best_len:
            best = list(chosen_set)
            best_len = len(chosen)
        return chosen_set

    # Strategy 1: Greedy construction with multiple orderings
    # Order by "difficulty" or just try different permutations
    # A good heuristic: try elements in increasing order of value (small reciprocals first? No, large reciprocals are "heavier" in integer space)
    # Actually, 1/1 is the largest reciprocal. 1/22 is the smallest.
    # In integer space, L/1 is huge, L/22 is small.
    # Collisions are more likely with large integers?
    # Let's try: 1) Descending index (current), 2) Ascending index, 3) Randomized
    
    # The current best is 9. Can we get 10, 11, 12?
    # Let's run a DFS with better pruning and ordering.
    
    # Improved DFS: 
    # 1. Order variables by "impact". 
    # 2. Use bitsets or efficient sets.
    
    # Let's try a smarter DFS that picks the next element dynamically based on which ones are "compatible" with the current set.
    # This is expensive. Let's stick to static order but try multiple static orders.
    
    orders = [
        list(range(22, 0, -1)),  # 22, 21, ... 1
        list(range(1, 23)),      # 1, 2, ... 22
        # Sort by reciprocal size (ascending: small integers first)
        sorted(range(1, 23), key=lambda x: recip[x]),
        # Sort by reciprocal size (descending: large integers first)
        sorted(range(1, 23), key=lambda x: -recip[x]),
    ]

    for order in orders:
        if time.time() > deadline - 5.0:
            break
            
        def dfs(idx, sums, chosen):
            nonlocal best, best_len
            if time.time() > deadline - 1.0:
                return
            # prune: even adding all remaining can't beat best
            remaining = len(order) - idx
            if len(chosen) + remaining <= best_len:
                return
            if idx == len(order):
                if len(chosen) > best_len:
                    best = list(chosen)
                    best_len = len(chosen)
                return
            i = order[idx]
            r = recip[i]
            # try include
            if can_add(sums, r):
                new_sums = sums | {s + r for s in sums}
                chosen.append(i)
                dfs(idx + 1, new_sums, chosen)
                chosen.pop()
            # try exclude
            dfs(idx + 1, sums, chosen)

        dfs(0, {0}, [])
        
        # After DFS, try local search on the best found
        if time.time() < deadline - 3.0:
            local_search(best)

    return best
