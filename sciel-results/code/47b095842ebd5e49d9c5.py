"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from random import Random

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_sum_free_local_search(elements):
        """
        Uses a local search heuristic starting from the largest absolute values.
        1. Sort by abs value descending.
        2. Take top k.
        3. Remove elements causing conflicts (smallest abs value first).
        4. Try to add back remaining elements.
        """
        n = len(elements)
        if n == 0:
            return []
        
        # Sort by absolute value descending
        sorted_elems = sorted(elements, key=lambda x: -abs(x))
        
        # Start with a large candidate set
        # We aim for a size slightly larger than n/2, as n/2 is a guaranteed lower bound
        initial_k = max(1, int(n * 0.6))
        current_set_list = sorted_elems[:initial_k]
        
        # Repair phase: Remove elements that cause sum-free violations
        changed = True
        while changed:
            changed = False
            current_set_set = set(current_set_list)
            # Find a violation
            violation = None
            for i in range(len(current_set_list)):
                a = current_set_list[i]
                for j in range(i, len(current_set_list)):
                    b = current_set_list[j]
                    s = a + b
                    if s in current_set_set:
                        violation = (a, b, s)
                        break
                if violation:
                    break
            
            if violation:
                a, b, c = violation
                # Remove the element with the smallest absolute value
                candidates = [a, b, c]
                min_abs_elem = min(candidates, key=lambda x: abs(x))
                if min_abs_elem in current_set_list:
                    current_set_list.remove(min_abs_elem)
                    changed = True
        
        # Improvement phase: Try to add back elements
        current_set_set = set(current_set_list)
        remaining = [x for x in sorted_elems if x not in current_set_set]
        
        for x in remaining:
            # Check if adding x creates a violation
            can_add = True
            
            # Check 1: x + y in S for any y in S
            for y in current_set_list:
                if x + y in current_set_set:
                    can_add = False
                    break
            
            # Check 2: x + x in S
            if can_add and (2 * x) in current_set_set:
                can_add = False
            
            # Check 3: a + b == x for any a, b in S
            if can_add:
                for i in range(len(current_set_list)):
                    a = current_set_list[i]
                    for j in range(i, len(current_set_list)):
                        b = current_set_list[j]
                        if a + b == x:
                            can_add = False
                            break
                    if not can_add:
                        break
            
            if can_add:
                current_set_list.append(x)
                current_set_set.add(x)
                
        return current_set_list

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = solve_sum_free_local_search(elems)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
