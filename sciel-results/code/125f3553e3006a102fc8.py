"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))
    
    # Precompute reciprocal as integer: 1/i * L
    recip = [L // i for i in range(1, 23)]
    
    # Best known fallback: 9 elements
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    def is_dissociated(selected):
        """Check if all subset sums are distinct using integer arithmetic."""
        if not selected:
            return True
        n = len(selected)
        sums = [0]
        for idx in selected:
            r = recip[idx - 1]
            new_sums = [s + r for s in sums]
            sums.extend(new_sums)
        # Check all sums are distinct
        return len(sums) == len(set(sums))
    
    def incremental_check(current_sums_set, current_sums_list, new_idx):
        """
        Incrementally check if adding new_idx keeps all subset sums distinct.
        Returns (is_valid, new_sums_set, new_sums_list) or (False, None, None).
        """
        r = recip[new_idx - 1]
        # New sums are old_sums + r
        new_sums = [s + r for s in current_sums_list]
        # Check if any new sum already exists in current_sums_set
        # Also check if any new sums collide with each other (they won't if old were distinct and we just add r)
        # Actually, the only collisions possible are between new_sums and old_sums.
        for s in new_sums:
            if s in current_sums_set:
                return False, None, None
        
        # Merge
        new_set = current_sums_set.union(new_sums)
        new_list = current_sums_list + new_sums
        return True, new_set, new_list
    
    def backtrack(idx, current, current_sums_set, current_sums_list):
        """Backtracking search for maximum dissociated subset."""
        nonlocal best_result
        if time.time() > deadline:
            return
        
        # Prune: if we can't beat current best
        remaining = 22 - idx + 1
        if len(current) + remaining <= len(best_result):
            return
        
        # Try including idx first (to find large sets faster)
        if idx <= 22:
            valid, new_set, new_list = incremental_check(current_sums_set, current_sums_list, idx)
            if valid:
                new_current = current + [idx]
                if len(new_current) > len(best_result):
                    best_result = new_current[:]
                backtrack(idx + 1, new_current, new_set, new_list)
        
        # Try not including idx
        backtrack(idx + 1, current, current_sums_set, current_sums_list)
    
    # Start backtracking search
    # Initial state: empty set
    initial_sums_set = {0}
    initial_sums_list = [0]
    
    # We want to maximize cardinality.
    # Backtracking with "include first" tends to find large sets quickly.
    # The pruning `len(current) + remaining <= len(best_result)` is effective.
    
    backtrack(1, [], initial_sums_set, initial_sums_list)
    
    return best_result
# EVOLVE-BLOCK-END
