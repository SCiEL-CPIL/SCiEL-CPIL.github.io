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

    def dfs(idx, order, sums, chosen):
        nonlocal best, best_len
        if time.time() > deadline:
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
        rest = order[idx + 1:]
        # try include first
        if can_add(sums, r):
            new_sums = sums | {s + r for s in sums}
            chosen.append(i)
            dfs(idx + 1, order, new_sums, chosen)
            chosen.pop()
        # try exclude
        dfs(idx + 1, order, sums, chosen)

    # Try multiple orderings to find larger sets
    # Strategy 1: descending order (smaller reciprocals first)
    dfs(0, list(range(22, 0, -1)), {0}, [])

    if time.time() > deadline:
        return best

    # Strategy 2: ascending order (larger reciprocals first)
    dfs(0, list(range(1, 23)), {0}, [])

    if time.time() > deadline:
        return best

    # Strategy 3: alternate - pick elements that are "independent" 
    # Use a heuristic: sort by how "compatible" they are with each other
    # For now, try a few more permutations
    
    # Strategy 4: start with the known best and try to add more
    # We can do a local search: try to add elements one by one
    
    # Let's try a greedy approach with different starting points
    for start_elem in range(1, 23):
        if time.time() > deadline:
            break
        # Try to build a set starting with start_elem
        sums = {0, recip[start_elem]}
        chosen = [start_elem]
        
        # Try adding other elements
        for i in range(1, 23):
            if time.time() > deadline:
                break
            if i == start_elem:
                continue
            if can_add(sums, recip[i]):
                new_sums = sums | {s + recip[i] for s in sums}
                chosen.append(i)
                sums = new_sums
                if len(chosen) > best_len:
                    best = list(chosen)
                    best_len = len(chosen)
    
    return best
