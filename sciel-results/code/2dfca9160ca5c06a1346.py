"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 256
    
    # Start with the proven interval [171, 256]
    # For a in [171,256], b,c in [171,256] with a<b<=c:
    # b+c >= (a+1) + (a+1) = 2a+2
    # Next multiple is 3a. For a >= 171, 3a >= 513 > 512 >= b+c.
    # So for a >= 171, no multiple of a in [2a+2, 512].
    # Thus [171, 256] is feasible.
    
    base_set = list(range(171, n + 1))
    selected = set(base_set)
    sorted_sel = sorted(selected)
    
    def is_feasible(candidate, sel_set, sorted_sel_list):
        """Check if adding candidate to sel_set maintains the property."""
        # Case 1: candidate is 'a' (smallest)
        # For all b,c in sel_set with b > candidate, c >= b: candidate ∤ (b+c)
        for i, b in enumerate(sorted_sel_list):
            if b <= candidate:
                continue
            for j in range(i, len(sorted_sel_list)):
                c = sorted_sel_list[j]
                if c < b:
                    continue
                if (b + c) % candidate == 0:
                    return False
        # Case 2: candidate is 'b' or 'c'
        # For all a in sel_set with a < candidate
        for a in sorted_sel_list:
            if a >= candidate:
                continue
            # Sub-case 2a: a < candidate <= b, check a ∤ (candidate + b)
            for b in sorted_sel_list:
                if b < candidate:
                    continue
                if (candidate + b) % a == 0:
                    return False
            # Sub-case 2b: a < b < candidate, check a ∤ (b + candidate)
            for b in sorted_sel_list:
                if b >= candidate:
                    continue
                if (b + candidate) % a == 0:
                    return False
        # Sub-case 2c: candidate = b = c, a < candidate, a ∤ (2*candidate)
        for a in sorted_sel_list:
            if a >= candidate:
                continue
            if (2 * candidate) % a == 0:
                return False
        return True
    
    def try_add(num, selected, sorted_sel):
        """Try to add num to selected. Returns True if successful."""
        if num in selected:
            return True
        if is_feasible(num, selected, sorted_sel):
            selected.add(num)
            # Re-sort
            return True
        return False
    
    # Greedily try to add numbers from small to large
    for num in range(1, 171):
        if time.time() > deadline:
            break
        if try_add(num, selected, sorted_sel):
            sorted_sel = sorted(selected)
    
    # Local search: try to swap elements to improve
    # Try removing one element and adding two
    improvement_found = True
    max_improvement_iterations = 50
    iteration = 0
    
    while improvement_found and iteration < max_improvement_iterations:
        if time.time() > deadline - 5:
            break
        improvement_found = False
        iteration += 1
        
        # Try removing a random element and adding a smaller one
        if len(selected) > 100:
            elements = list(selected)
            random.shuffle(elements)
            for remove_elem in elements[:20]:
                if time.time() > deadline - 3:
                    break
                # Try to remove remove_elem and add a smaller number
                for add_elem in range(1, remove_elem):
                    if add_elem in selected:
                        continue
                    # Check if removing remove_elem and adding add_elem is feasible
                    temp_selected = selected - {remove_elem}
                    temp_sorted = sorted(temp_selected)
                    if is_feasible(add_elem, temp_selected, temp_sorted):
                        # Check that the new set is still valid (it should be by construction)
                        selected = temp_selected
                        selected.add(add_elem)
                        sorted_sel = sorted(selected)
                        improvement_found = True
                        break
                if improvement_found:
                    break
    
    result = sorted(selected)
    return result

# EVOLVE-BLOCK-END
