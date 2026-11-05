"""Improved program for 3-AP-free subset maximizing reciprocal sum."""
# EVOLVE-BLOCK-START
import time

def _has_3ap(selected_set):
    """Check if selected_set contains any 3-term AP."""
    s = set(selected_set)
    for a in s:
        for b in s:
            if a < b:
                c = 2*b - a
                if c in s and c != a and c != b:
                    return True
    return False

def _is_3ap_free(selected):
    """Check if a list has no 3-term AP."""
    s = set(selected)
    for a in s:
        for b in s:
            if a < b:
                c = 2*b - a
                if c in s and c != a and c != b:
                    return False
    return True

def _reciprocal_sum(selected):
    """Compute reciprocal sum as float."""
    return sum(1.0/x for x in selected)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Phase 1: Greedy construction (same as seed)
    selected = []
    for x in range(1, 257):
        ok = True
        for a in selected:
            for b in selected:
                if a < b:
                    c = 2*b - a
                    if c == x:
                        ok = False
                        break
                if b < a:
                    c = 2*a - b
                    if c == x:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            selected.append(x)
    
    best_set = set(selected)
    best_sum = _reciprocal_sum(selected)
    
    # Phase 2: Local search - try removals and re-additions
    # Try removing one element and see if we can add better ones
    for _round in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each element for removal
        for rem in sorted(best_set, key=lambda x: 1.0/x, reverse=True):  # try removing high-value first
            if time.time() > deadline:
                break
            
            trial = best_set - {rem}
            trial_sum = best_sum - 1.0/rem
            
            # Try adding any element not in trial
            for x in range(1, 257):
                if x in trial:
                    continue
                # Check if adding x keeps it 3-AP-free
                ok = True
                for a in trial:
                    for b in trial:
                        if a < b:
                            c = 2*b - a
                            if c == x:
                                ok = False
                                break
                        if b < a:
                            c = 2*a - b
                            if c == x:
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    new_sum = trial_sum + 1.0/x
                    if new_sum > best_sum + 1e-12:
                        best_set = trial | {x}
                        best_sum = new_sum
                        improved = True
                        break  # restart with new best
    
        if not improved:
            break
    
    # Phase 3: Try swap operations - remove one, add another
    for _round in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for rem in list(best_set):
            if time.time() > deadline:
                break
            
            trial = best_set - {rem}
            
            for add in range(1, 257):
                if add in trial:
                    continue
                # Check if adding add to trial is 3-AP-free
                ok = True
                for a in trial:
                    for b in trial:
                        if a < b:
                            c = 2*b - a
                            if c == add:
                                ok = False
                                break
                        if b < a:
                            c = 2*a - b
                            if c == add:
                                ok = False
                                break
                    if not ok:
                        break
                
                if ok:
                    new_sum = best_sum - 1.0/rem + 1.0/add
                    if new_sum > best_sum + 1e-12:
                        best_set = trial | {add}
                        best_sum = new_sum
                        improved = True
                        break
    
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
