"""Rewritten program: branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
import random
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    # Order candidates: Ascending value (1, 2, ..., 22)
    # This corresponds to Descending Numerator (L, L/2, ..., L/22)
    # Starting with large steps keeps the sum set sparse, reducing collisions.
    order = list(range(N))  # indices for elements 1, 2, ..., 22
    vals = [universe[i] for i in order]
    nums = [num[i] for i in order]

    # Greedy initialization to set a strong initial bound
    def _greedy_solve(ord_idx):
        cur_sums = {0}
        cur_list = []
        for i in ord_idx:
            n = nums[i]
            v = vals[i]
            # Check if we can add n
            if cur_sums.isdisjoint(s + n for s in cur_sums):
                new_sums = cur_sums | {s + n for s in cur_sums}
                cur_sums = new_sums
                cur_list.append(v)
        return cur_list

    # Multi-start greedy to find a better initial bound
    best = []
    best_len = 0
    
    # Define several orders to try
    orders_to_try = [
        list(range(N)),          # 1, 2, ..., 22
        list(range(N-1, -1, -1)), # 22, 21, ..., 1
        [i for i in range(N) if (i+1) % 2 != 0] + [i for i in range(N) if (i+1) % 2 == 0], # Odds then Evens
        [i for i in range(N) if (i+1) % 2 == 0] + [i for i in range(N) if (i+1) % 2 != 0], # Evens then Odds
    ]
    
    # Add a few random shuffles
    rng = random.Random(42)
    for _ in range(10):
        shuffled = list(range(N))
        rng.shuffle(shuffled)
        orders_to_try.append(shuffled)

    for ord_idx in orders_to_try:
        res = _greedy_solve(ord_idx)
        if len(res) > best_len:
            best_len = len(res)
            best = res

    def _time_ok():
        return (time.time() - start_time) < deadline

    # Iterative DFS
    # Stack items: (pos, cur_list, cur_sums)
    # We push 'exclude' then 'include' so that 'include' is processed first (LIFO)
    
    stack = [(0, [], {0})]
    
    while stack:
        pos, cur_list, cur_sums = stack.pop()
        
        if not _time_ok():
            break
            
        # Prune: even if we add all remaining, can't beat best
        remaining = N - pos
        if len(cur_list) + remaining <= best_len:
            continue
            
        if pos == N:
            if len(cur_list) > best_len:
                best_len = len(cur_list)
                best = cur_list[:]
            continue

        n = nums[pos]
        v = vals[pos]
        
        # Check collision for including n
        # Collision if (s + n) in cur_sums for any s in cur_sums
        # Using isdisjoint with a generator is efficient
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

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
