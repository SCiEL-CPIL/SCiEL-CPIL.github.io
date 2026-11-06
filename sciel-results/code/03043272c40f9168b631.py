import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    def has_3ap_with_x(selected_set, x):
        """Check if adding x to selected_set creates a 3-AP."""
        s = selected_set
        # x is middle: a + b = 2x for a,b in s, a != b
        two_x = 2 * x
        for a in s:
            b = two_x - a
            if b != a and b in s:
                return True
        # x is max: x = 2b - a for a < b in s
        for b in s:
            a = 2 * b - x
            if a < b and a in s:
                return True
        # x is min: x = 2a - b for a < b in s, so x < a
        for a in s:
            b = 2 * a - x
            if a < b and b in s and x < a:
                return True
        return False
    
    def is_3ap_free(selected_list):
        s = set(selected_list)
        for x in s:
            if has_3ap_with_x(s - {x}, x):
                return False
        return True
    
    # Greedy initial: try to add small numbers first
    selected = set()
    for x in range(1, 257):
        if not has_3ap_with_x(selected, x):
            selected.add(x)
    
    best_set = set(selected)
    best_sum = sum(Fraction(1, x) for x in best_set)
    
    # Local search
    all_candidates = set(range(1, 257))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in sorted(all_candidates - best_set, key=lambda v: -1/v):
            if time.time() >= deadline:
                break
            if not has_3ap_with_x(best_set, x):
                new_set = best_set | {x}
                new_sum = best_sum + Fraction(1, x)
                if new_sum > best_sum:
                    best_set = new_set
                    best_sum = new_sum
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        for remove_x in sorted(best_set, key=lambda v: 1/v):  # remove small reciprocal first
            if time.time() >= deadline:
                break
            for add_x in sorted(all_candidates - best_set, key=lambda v: -1/v):
                if time.time() >= deadline:
                    break
                if add_x == remove_x:
                    continue
                new_set = (best_set - {remove_x}) | {add_x}
                if not has_3ap_with_x(new_set - {add_x}, add_x):
                    new_sum = best_sum - Fraction(1, remove_x) + Fraction(1, add_x)
                    if new_sum > best_sum:
                        best_set = new_set
                        best_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try 2-swap: remove two, add two (more expensive)
            found = False
            items = list(best_set)
            for i in range(len(items)):
                if time.time() >= deadline:
                    break
                for j in range(i+1, len(items)):
                    if time.time() >= deadline:
                        break
                    rem1, rem2 = items[i], items[j]
                    temp_set = best_set - {rem1, rem2}
                    for add1 in sorted(all_candidates - temp_set, key=lambda v: -1/v):
                        if time.time() >= deadline:
                            break
                        if add1 in temp_set:
                            continue
                        if not has_3ap_with_x(temp_set, add1):
                            temp_set2 = temp_set | {add1}
                            for add2 in sorted(all_candidates - temp_set2, key=lambda v: -1/v):
                                if time.time() >= deadline:
                                    break
                                if add2 in temp_set2:
                                    continue
                                if not has_3ap_with_x(temp_set2, add2):
                                    new_set = temp_set2 | {add2}
                                    new_sum = best_sum - Fraction(1, rem1) - Fraction(1, rem2) + Fraction(1, add1) + Fraction(1, add2)
                                    if new_sum > best_sum:
                                        best_set = new_set
                                        best_sum = new_sum
                                        found = True
                                        break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            else:
                break
    
    return sorted(best_set)
