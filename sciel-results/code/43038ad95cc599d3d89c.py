"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    if n > 20:
        # Too large for exact check within time budget
        return False
    
    # Calculate LCM
    L = reduce(math.lcm, elems)
    
    # Convert to scaled integers
    weights = [L // e for e in elems]
    
    sums = {0}
    for w in weights:
        # Check for collision: if any s + w is already in sums
        # We can do this more efficiently by checking intersection
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = sums | new_sums
        
    return len(sums) == (1 << n)

def _can_add(elems, new_elem):
    """Check if new_elem can be added to elems without creating collisions."""
    if not elems:
        return True
    
    test_elems = list(elems) + [new_elem]
    n = len(test_elems)
    if n > 20:
        return False
    
    L = reduce(math.lcm, test_elems)
    weights = [L // e for e in test_elems]
    
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums = sums | new_sums
        
    return len(sums) == (1 << n)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Greedy function with a given order
    def greedy(order):
        cur = []
        for e in order:
            if time.time() > deadline - 10:
                break
            test_set = cur + [e]
            if _is_dissociated(test_set):
                cur = test_set
        return cur

    best_set = []
    
    # 1. Try deterministic orders - multiple heuristics
    orders = [
        sorted(universe, reverse=True),  # Largest first (smallest reciprocals)
        sorted(universe),                 # Smallest first
        # Primes first, then composites
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        # Largest primes first
        [19, 17, 13, 11, 7, 5, 3, 2, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4],
        # Interleaved large-small
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        # Reverse interleaved
        [11, 12, 10, 13, 9, 14, 8, 15, 7, 16, 6, 17, 5, 18, 4, 19, 3, 20, 2, 21, 1, 22],
    ]
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result[:]
    
    # 2. Local search: try to add any missing element
    def local_search_add(s, time_limit=5):
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            remaining = [e for e in universe if e not in s]
            random.shuffle(remaining)
            for e in remaining:
                test_set = s + [e]
                if _is_dissociated(test_set):
                    s = test_set
                    improved = True
                    break
        return s

    if best_set:
        best_set = local_search_add(best_set, 10)
    
    # 3. Random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 20:
        random.shuffle(universe)
        result = greedy(universe)
        result = local_search_add(result, 5)
        
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= 18:
                break
        num_restarts += 1
    
    # 4. Local search with swaps to escape local maxima
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        s_set = set(best_set)
        in_set = list(best_set)
        out_set = [e for e in universe if e not in s_set]
        random.shuffle(out_set)
        
        for u in in_set:
            if time.time() > deadline - 10:
                break
            for v in out_set:
                if time.time() > deadline - 10:
                    break
                # Try swap
                new_s = [x for x in best_set if x != u] + [v]
                if _is_dissociated(new_s):
                    # Check if this swap allows adding more
                    extended = local_search_add(new_s, 3)
                    if len(extended) > len(best_set):
                        best_set = extended
                        improved = True
                        break
            if improved:
                break
    
    # 5. Final aggressive local search
    if best_set and time.time() < deadline - 5:
        best_set = local_search_add(best_set, 5)
    
    # Final verification
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback to a known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
