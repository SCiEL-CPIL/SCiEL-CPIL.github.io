"""Rewritten program: branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    best = []
    best_len = 0

    def _time_ok():
        return (time.time() - start_time) < deadline

    def _dfs(order):
        nonlocal best, best_len
        
        # Iterative DFS
        # Stack items: (pos, cur_list, cur_sums)
        stack = [(0, [], {0})]
        
        while stack:
            pos, cur_list, cur_sums = stack.pop()
            
            if not _time_ok():
                return
                
            # Prune: even if we add all remaining, can't beat best
            remaining = N - pos
            if len(cur_list) + remaining <= best_len:
                continue
                
            if pos == N:
                if len(cur_list) > best_len:
                    best_len = len(cur_list)
                    best = cur_list[:]
                continue

            idx = order[pos]
            n = num[idx]
            v = universe[idx]
            
            # Check collision for including n
            # Collision if (s + n) in cur_sums for any s in cur_sums
            if cur_sums.isdisjoint(s + n for s in cur_sums):
                # No collision, we can include
                new_sums = cur_sums | {s + n for s in cur_sums}
                new_list = cur_list + [v]
                # Push exclude first so include is popped first
                stack.append((pos + 1, cur_list, cur_sums))
                stack.append((pos + 1, new_list, new_sums))
            else:
                # Collision, must exclude
                stack.append((pos + 1, cur_list, cur_sums))

    # Try multiple orderings to increase chance of finding a larger set
    # Order 1: Ascending denominator (1, 2, ..., 22) -> Descending reciprocal value
    order1 = list(range(N))
    _dfs(order1)
    
    # Order 2: Descending denominator (22, 21, ..., 1) -> Ascending reciprocal value
    if _time_ok():
        order2 = list(range(N-1, -1, -1))
        _dfs(order2)

    # Order 3: Randomized (optional, if time remains)
    if _time_ok():
        import random
        order3 = list(range(N))
        random.shuffle(order3)
        _dfs(order3)

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
