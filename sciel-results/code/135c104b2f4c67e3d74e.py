import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to 512
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Filter: remove numbers where 2*a is a perfect square
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
    
    # Build conflict graph
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
    
    def greedy_select(cand_list, conflict_map, seed=None):
        """Greedy: repeatedly pick vertex with min degree, remove it and neighbors."""
        if seed is not None:
            random.seed(seed)
        remaining = set(cand_list)
        result = []
        while remaining:
            min_deg = float('inf')
            best = None
            for v in remaining:
                deg = len(conflict_map[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best = v
            if best is None:
                break
            result.append(best)
            to_remove = {best} | (conflict_map[best] & remaining)
            remaining -= to_remove
        return result
    
    def local_search(subset, conflict_map, cand_list, max_iter=5000):
        """Improve subset by adding non-conflicting elements or swapping."""
        s = set(subset)
        improved = True
        iterations = 0
        cand_set = set(cand_list)
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            # Try adding
            for a in cand_set - s:
                if conflict_map[a] & s:
                    continue
                s.add(a)
                improved = True
                break
            if improved:
                continue
            # Try swapping: remove one, add one
            found_swap = False
            for remove_v in list(s):
                new_s = s - {remove_v}
                for add_v in cand_set - s:
                    if conflict_map[add_v] & new_s:
                        continue
                    s = new_s | {add_v}
                    found_swap = True
                    break
                if found_swap:
                    break
            if found_swap:
                improved = True
        return s
    
    # Initial greedy
    best_set = greedy_select(candidates, conflict, seed=42)
    best_size = len(best_set)
    
    # Local search
    improved_set = local_search(best_set, conflict, candidates)
    if len(improved_set) > best_size:
        best_set = list(improved_set)
        best_size = len(best_set)
    
    # Random restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        for seed in range(1, 200):
            if time.time() - start_time > deadline - 5:
                break
            gs = greedy_select(candidates, conflict, seed=seed)
            ls = local_search(gs, conflict, candidates, max_iter=2000)
            if len(ls) > best_size:
                best_set = list(ls)
                best_size = len(ls)
    
    # Final local search passes
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        for _ in range(30):
            if time.time() - start_time > deadline - 3:
                break
            improved = local_search(best_set, conflict, candidates, max_iter=3000)
            if len(improved) > best_size:
                best_set = list(improved)
                best_size = len(improved)
    
    return best_set
