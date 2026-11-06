"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 512
    
    # Step 1: Base-3 digit-based construction (Cantor-like set)
    # Select numbers in [1, n] whose base-3 representation contains no digit '2'
    # This set is 3-AP-free, and therefore 4-AP-free.
    present = np.zeros(n + 1, dtype=bool)
    for x in range(1, n + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 2:
                valid = False
                break
            num //= 3
        if valid:
            present[x] = True
            
    # Helper function to check if adding x creates a 4-AP
    # Returns True if x can be added safely (i.e., no 4-AP is formed)
    def can_add(x, p):
        # A 4-AP is a, a+d, a+2d, a+3d
        # If we add x, it might be any of the 4 terms.
        # We need to check if there exist d>0 such that the other 3 terms are in p.
        
        # Case 1: x is the 1st term (a). Terms: x, x+d, x+2d, x+3d
        # d can range from 1 to (n-x)//3
        # We need p[x+d] & p[x+2d] & p[x+3d] to be True for any d.
        # Vectorized check:
        # d = 1..max_d
        # Indices: x+1..x+max_d, x+2..x+2*max_d, x+3..x+3*max_d
        # This is tricky to vectorize directly without creating large arrays.
        # However, n=512 is small. We can loop over d but use numpy slicing for the checks?
        # Or just simple loops are fast enough if we are smart.
        # Let's try a simple loop for d, but check all 4 cases in one go?
        # No, the cases are distinct.
        
        # Optimized check:
        # Precompute max_d for each case?
        
        # Case 1: x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        if max_d1 > 0:
            # Check if any d in [1, max_d1] satisfies p[x+d] & p[x+2d] & p[x+3d]
            # We can construct arrays of indices
            ds = np.arange(1, max_d1 + 1)
            if np.any(p[x+ds] & p[x+2*ds] & p[x+3*ds]):
                return False

        # Case 2: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        if max_d2 > 0:
            ds = np.arange(1, max_d2 + 1)
            if np.any(p[x-ds] & p[x+ds] & p[x+2*ds]):
                return False

        # Case 3: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        if max_d3 > 0:
            ds = np.arange(1, max_d3 + 1)
            if np.any(p[x-2*ds] & p[x-ds] & p[x+ds]):
                return False

        # Case 4: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        if max_d4 > 0:
            ds = np.arange(1, max_d4 + 1)
            if np.any(p[x-3*ds] & p[x-2*ds] & p[x-ds]):
                return False

        return True

    # Local Search
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Try to add any missing element
        for x in range(1, n + 1):
            if not present[x]:
                if can_add(x, present):
                    present[x] = True
                    improved = True
                    # We could break here and restart, or continue scanning.
                    # Continuing scanning is fine, but might miss better orderings.
                    # Let's continue to find all immediate additions.
        
        # Phase 2: Try swaps (remove one, add one or more)
        # To maximize cardinality, we look for a swap that increases size.
        # Simple heuristic: For each element y in the set, remove it.
        # Then try to add any missing elements.
        # If count increases, keep it.
        
        # This phase is expensive if done naively.
        # Optimization: Only try to remove elements that are "critical" or just try all.
        # With n=512, and set size ~100-200, trying all removals and then greedy add is feasible.
        
        current_size = np.sum(present)
        
        for y in range(1, n + 1):
            if present[y]:
                # Temporarily remove y
                present[y] = False
                
                # Try to add missing elements
                added = 0
                added_elements = []
                for x in range(1, n + 1):
                    if not present[x]:
                        if can_add(x, present):
                            present[x] = True
                            added += 1
                            added_elements.append(x)
                
                new_size = np.sum(present)
                
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                    # Keep the change
                else:
                    # Revert
                    for x in added_elements:
                        present[x] = False
                    present[y] = True
                    # We don't break, we try other y's, but we must be careful
                    # that the state is consistent. Since we reverted, it is.
                    
    result = [x for x in range(1, n + 1) if present[x]]
    return sorted(result)
# EVOLVE-BLOCK-END
