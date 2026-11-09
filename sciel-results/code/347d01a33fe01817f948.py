"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    # Try to find the largest dissociated subset of [1, 22]
    # A set is dissociated if all 2^k subset sums of reciprocals are distinct
    
    def is_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        if len(elements) == 0:
            return True
        if len(elements) > 14:  # 2^14 = 16384, still feasible but slow
            # Use a heuristic: if we can't verify quickly, assume not
            # Actually let's try with a timeout
            pass
        
        k = len(elements)
        if k > 15:
            return False  # Too many to verify in reasonable time
        
        # Generate all subset sums
        sums = set()
        n = len(elements)
        for mask in range(1 << n):
            s = 0.0
            for i in range(n):
                if mask & (1 << i):
                    s += 1.0 / elements[i]
            # Use a small tolerance for floating point
            # Actually, for exactness, let's use fractions
            # But fractions might be slow. Let's use a different approach.
            sums.add(round(s, 10))
        
        return len(sums) == (1 << n)
    
    def is_dissociated_exact(elements):
        """Check dissociation using integer arithmetic with LCM scaling."""
        n = len(elements)
        if n == 0:
            return True
        if n > 15:
            return False
        
        # Compute LCM of all elements
        l = 1
        for e in elements:
            l = lcm(l, e)
        
        # Convert to integer weights: w_i = LCM / element_i
        weights = [l // e for e in elements]
        
        # Generate all subset sums
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += weights[i]
            sums.add(s)
        
        return len(sums) == (1 << n)
    
    # Greedy construction: try to build up a dissociated set
    # Try multiple orderings and strategies
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # Current known good set
    best_size = len(best_set)
    
    # Strategy 1: Greedy - add elements in various orders
    all_elements = list(range(1, 23))
    
    # Try greedy with different orderings
    orderings = [
        sorted(all_elements),
        sorted(all_elements, reverse=True),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    ]
    
    for ordering in orderings:
        current_set = []
        for e in ordering:
            if e in current_set:
                continue
            new_set = current_set + [e]
            if is_dissociated_exact(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
        
        if time.time() > deadline:
            break
    
    # Strategy 2: Start from known good set and try to add elements
    base_sets = [
        [1, 2, 3, 5, 7, 11, 13, 17, 19],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23] if 23 <= 22 else [1, 2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    for base in base_sets:
        base = [e for e in base if 1 <= e <= 22]
        current_set = list(base)
        for e in all_elements:
            if e in current_set:
                continue
            new_set = current_set + [e]
            if is_dissociated_exact(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
        
        if time.time() > deadline:
            break
    
    # Strategy 3: Randomized local search
    import random
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random subset
        current_set = []
        for e in random.sample(all_elements, 10):
            new_set = current_set + [e]
            if is_dissociated_exact(new_set):
                current_set = new_set
        
        # Try to add more elements
        for e in all_elements:
            if e in current_set:
                continue
            new_set = current_set + [e]
            if is_dissociated_exact(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Strategy 4: Try all pairs to swap out one element for another
    if best_size < 12:
        for i, e_out in enumerate(best_set):
            if time.time() > deadline:
                break
            for e_in in all_elements:
                if e_in in best_set:
                    continue
                new_set = best_set[:i] + best_set[i+1:] + [e_in]
                if is_dissociated_exact(new_set):
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = new_set
                    # Try to add more after swap
                    for e_add in all_elements:
                        if e_add in new_set:
                            continue
                        new_set2 = new_set + [e_add]
                        if is_dissociated_exact(new_set2):
                            new_set = new_set2
                            if len(new_set) > best_size:
                                best_size = len(new_set)
                                best_set = new_set
    
    # Ensure we return a nonempty valid set
    if not best_set or len(best_set) == 0:
        best_set = [1]
    
    # Final verification
    if not is_dissociated_exact(best_set):
        # Fallback to known good set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        if not is_dissociated_exact(best_set):
            best_set = [1]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
