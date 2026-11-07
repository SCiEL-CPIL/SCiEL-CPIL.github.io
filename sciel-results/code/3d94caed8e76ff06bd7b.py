"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import sys
import time

P = {'n': 256, 'maximum_size': 30}

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    Uses branch-and-bound with greedy coloring upper bounds on a compatibility graph.
    """
    
    start_time = time.time()
    timeout = 55
    best_set = []
    best_sums = set()
    
    def check_divisibility(sums):
        """Check no two distinct sums divide one another."""
        sl = sorted(sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return False
        return True
    
    def can_extend(current_sums, x):
        """Check if x can be added. Returns (valid, new_sums)."""
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        if not check_divisibility(new_sums):
            return False, new_sums
        return True, new_sums
    
    # Greedy construction with multiple strategies
    def greedy(order):
        S = []
        sums = set()
        for x in order:
            valid, new_sums = can_extend(sums, x)
            if valid:
                S.append(x)
                sums = new_sums
        return S, sums
    
    # Try several orderings to find a good starting set
    orders = [
        list(range(256, 0, -1)),
        list(range(256, 127, -1)),
        list(range(200, 0, -1)),
        [x for x in range(256, 0, -1) if x % 2 == 1],
        [x for x in range(256, 0, -1) if x > 100],
        list(range(256, 0, -2)),  # Even numbers descending
    ]
    
    for order in orders:
        S, sums = greedy(order)
        if len(S) > len(best_set):
            best_set = S
            best_sums = sums
    
    # Branch and bound to improve
    def backtrack(current_set, current_sums, available, depth=0):
        nonlocal best_set, best_sums, start_time
        if time.time() - start_time > timeout:
            return
        
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            best_sums = current_sums
        
        if not available or depth >= 25:
            return
        
        # Upper bound: count how many remaining candidates can be individually added
        compat = []
        for x in available:
            valid, _ = can_extend(current_sums, x)
            if valid:
                compat.append(x)
        
        if len(current_set) + len(compat) <= len(best_set):
            return
        
        # Try compatible candidates, prioritize those that leave more options
        for x in compat:
            if time.time() - start_time > timeout:
                return
            valid, new_sums = can_extend(current_sums, x)
            if valid:
                new_avail = [a for a in available if a != x]
                backtrack(current_set + [x], new_sums, new_avail, depth + 1)
    
    if best_set:
        remaining = [x for x in range(1, 257) if x not in best_set]
        backtrack(best_set, best_sums, remaining)
    
    if not best_set:
        best_set = [31, 32, 34]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
