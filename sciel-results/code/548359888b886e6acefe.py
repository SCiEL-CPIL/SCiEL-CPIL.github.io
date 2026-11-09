"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def can_add(selected, triple_sums, x):
        """Check if x can be added to selected without creating duplicate triple sums."""
        n = len(selected)
        # New triples involving x:
        # (x, x, x) -> 3x
        # (x, x, y) for y in selected -> 2x + y
        # (x, y, z) for y <= z in selected -> x + y + z
        
        # Check 3x
        s = 3 * x
        if s in triple_sums:
            return False
        
        # Check 2x + y for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in triple_sums:
                return False
        
        # Check x + y + z for y <= z in selected
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in triple_sums:
                    return False
        
        return True

    def add_element(selected, triple_sums, x):
        """Add x to selected and update triple_sums."""
        n = len(selected)
        triple_sums.add(3 * x)
        for y in selected:
            triple_sums.add(2 * x + y)
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                triple_sums.add(x + y + z)
        selected.append(x)

    def build_greedy(start_from=1, max_size=MAX_SIZE, order='ascending'):
        """Greedy construction with specified ordering."""
        selected = []
        triple_sums = set()
        
        if order == 'ascending':
            candidates = range(start_from, N + 1)
        elif order == 'descending':
            candidates = range(N, start_from - 1, -1)
        elif order == 'power2':
            # Try powers of 2 first, then fill in
            candidates = sorted(range(start_from, N + 1), key=lambda x: (not (x & (x-1) == 0), x))
        else:
            candidates = range(start_from, N + 1)
        
        for x in candidates:
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 15:
                break
            
            if can_add(selected, triple_sums, x):
                add_element(selected, triple_sums, x)
        
        return sorted(selected)

    def local_improve(selected, triple_sums, max_iterations=100):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        triple_sums = set(triple_sums)
        improved = True
        iters = 0
        
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            if time.time() > DEADLINE - 10:
                break
            
            n = len(selected)
            # Try removing one element and adding a different one
            for idx in range(n):
                if time.time() > DEADLINE - 10:
                    break
                removed = selected[idx]
                # Remove from triple_sums
                # Need to remove all triples involving removed
                new_sums = set()
                for t in triple_sums:
                    # Check if t can be formed without 'removed'
                    # This is expensive; instead, rebuild
                    pass
                
                # Rebuild triple_sums without the removed element
                temp_selected = selected[:idx] + selected[idx+1:]
                temp_sums = set()
                for t in combinations_with_replacement(temp_selected, 3):
                    temp_sums.add(sum(t))
                
                # Try to add a new element
                best_new = None
                for x in range(1, N + 1):
                    if x == removed:
                        continue
                    if x in temp_selected:
                        continue
                    if can_add(temp_selected, temp_sums, x):
                        # Check if this improves (e.g., allows more elements later)
                        # For now, just accept if it maintains size
                        best_new = x
                        break
                
                if best_new is not None:
                    # Apply the swap
                    selected = temp_selected + [best_new]
                    selected.sort()
                    triple_sums = set()
                    for t in combinations_with_replacement(selected, 3):
                        triple_sums.add(sum(t))
                    improved = True
                    break
        
        return selected

    # Strategy 1: Simple ascending greedy
    result1 = build_greedy(start_from=1, order='ascending')
    
    # Strategy 2: Try with different starting points or orderings
    best_result = result1
    
    # Try descending
    if time.time() < DEADLINE - 30:
        result2 = build_greedy(start_from=1, order='descending')
        if len(result2) > len(best_result):
            best_result = result2
    
    # Try power-of-2 first
    if time.time() < DEADLINE - 30:
        result3 = build_greedy(start_from=1, order='power2')
        if len(result3) > len(best_result):
            best_result = result3

    # Local improvement
    if time.time() < DEADLINE - 20:
        triple_sums = set()
        for t in combinations_with_replacement(best_result, 3):
            triple_sums.add(sum(t))
        improved = local_improve(best_result, triple_sums, max_iterations=50)
        if len(improved) > len(best_result):
            best_result = improved

    return best_result
# EVOLVE-BLOCK-END
