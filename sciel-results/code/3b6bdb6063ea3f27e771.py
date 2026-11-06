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
    
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # Fallback
    
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
    
    def backtrack(idx, current, best_size):
        """Backtracking search for maximum dissociated subset."""
        nonlocal best_result
        if time.time() > deadline:
            return
        
        # Prune: if we can't beat current best
        remaining = 22 - idx + 1
        if len(current) + remaining <= len(best_result):
            return
        
        # Try not including idx
        backtrack(idx + 1, current, best_size)
        
        # Try including idx
        if idx <= 22:
            new_current = current + [idx]
            if is_dissociated(new_current):
                if len(new_current) > len(best_result):
                    best_result = new_current[:]
                backtrack(idx + 1, new_current, len(new_current))
    
    # Search for maximum cardinality
    # Try sizes from large to small
    for target_size in range(22, len(best_result), -1):
        if time.time() > deadline:
            break
        
        # Use backtracking with pruning
        def search_fixed_size(idx, current, target_size):
            nonlocal best_result
            if time.time() > deadline:
                return False
            
            if len(current) == target_size:
                if is_dissociated(current):
                    best_result = current[:]
                    return True
                return False
            
            # Pruning: not enough elements left
            if len(current) + (22 - idx + 1) < target_size:
                return False
            
            # Try including idx
            if idx <= 22:
                new_current = current + [idx]
                if is_dissociated(new_current):
                    if search_fixed_size(idx + 1, new_current, target_size):
                        return True
            
            # Try excluding idx
            if search_fixed_size(idx + 1, current, target_size):
                return True
            
            return False
        
        if search_fixed_size(1, [], target_size):
            break
    
    return best_result
# EVOLVE-BLOCK-END
