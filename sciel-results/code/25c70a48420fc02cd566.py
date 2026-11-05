"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Start with the proven valid upper tail [171, 256]
    # For a in [171, 256], 3*a >= 513 > max(b+c)=512
    # Thus a cannot divide b+c for any b,c in the set with a < b <= c
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    def is_valid_set(s):
        """Check if set s satisfies the property: for every a<b<=c in s, a does NOT divide b+c."""
        elements = sorted(s)
        n = len(elements)
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                for k in range(j, n):
                    c = elements[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def get_conflicts(s, x):
        """For a set s and element x not in s, find pairs (b,c) with b<=c in s, x<b<=c, and x divides b+c."""
        conflicts = []
        elements = sorted(s)
        for i in range(len(elements)):
            b = elements[i]
            if b <= x:
                continue
            for j in range(i, len(elements)):
                c = elements[j]
                if (b + c) % x == 0:
                    conflicts.append((b, c))
        return conflicts
    
    def can_add_with_removals(s, x, max_removals):
        """Try to add x to s by removing at most max_removals elements to resolve all conflicts."""
        conflicts = get_conflicts(s, x)
        if not conflicts:
            return True, set()
        
        # Find minimum removals to break all conflict pairs
        # A removal of element e breaks all pairs containing e
        all_elements_in_conflicts = set()
        for b, c in conflicts:
            all_elements_in_conflicts.add(b)
            all_elements_in_conflicts.add(c)
        
        # Greedy: try removing 1 element first
        for e in all_elements_in_conflicts:
            new_s = s - {e}
            if not get_conflicts(new_s, x):
                return True, {e}
        
        # Try removing 2 elements
        elem_list = list(all_elements_in_conflicts)
        for i in range(len(elem_list)):
            for j in range(i+1, len(elem_list)):
                e1, e2 = elem_list[i], elem_list[j]
                new_s = s - {e1, e2}
                if not get_conflicts(new_s, x):
                    return True, {e1, e2}
        
        # Try removing 3 elements
        if max_removals >= 3:
            for i in range(len(elem_list)):
                for j in range(i+1, len(elem_list)):
                    for k in range(j+1, len(elem_list)):
                        e1, e2, e3 = elem_list[i], elem_list[j], elem_list[k]
                        new_s = s - {e1, e2, e3}
                        if not get_conflicts(new_s, x):
                            return True, {e1, e2, e3}
        
        return False, set()
    
    # Try to add smaller elements one by one, starting from the smallest
    # For each candidate x, check if we can add it by removing few elements
    for x in range(1, 171):
        if time.time() > deadline:
            break
        
        if x in best_set:
            continue
        
        # Check if we can add x with at most 2 removals (net gain or neutral)
        success, removals = can_add_with_removals(best_set, x, 2)
        if success and len(removals) <= 1:
            # Net gain: add x, remove <=1 element
            new_set = best_set | {x} - removals
            if is_valid_set(new_set):
                best_set = new_set
                best_size = len(best_set)
    
    # Second pass: try again with the updated set
    for x in range(1, 171):
        if time.time() > deadline:
            break
        
        if x in best_set:
            continue
        
        success, removals = can_add_with_removals(best_set, x, 2)
        if success and len(removals) <= 1:
            new_set = best_set | {x} - removals
            if is_valid_set(new_set):
                best_set = new_set
                best_size = len(best_set)
    
    return list(best_set)
# EVOLVE-BLOCK-END
