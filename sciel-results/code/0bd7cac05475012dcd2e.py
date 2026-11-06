<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    
    # We want to find a set S of distinct positive integers <= 4096
    # such that all unordered triples (with repetition) have unique sums.
    # This is a B_3[1] sequence.
    
    # Strategy: Greedy with randomization and backtracking-lite
    # Key insight: Use numpy for fast sum collision checking
    # A triple sum a+b+c where a<=b<=c (unordered with repetition)
    
    # We'll try multiple random greedy attempts and keep the best
    
    best = []
    
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Random starting point to escape local optima
        start_val = random.randint(1, 50) if attempt > 0 else 1
        
        selected = []
        selected_sums = set()  # store all triple sums
        
        # We'll try to build the set greedily from start_val upward
        # But with some randomization in which candidates to try
        
        # Pre-compute: for a set S, adding x creates new triple sums:
        # x+x+x, x+x+y, x+y+y, x+y+z for all y,z in S
        # We need all these to be new (not in selected_sums)
        
        current = start_val
        max_val = n
        
        while current <= max_val:
            if time.time() > deadline - 5:
                break
            
            x = current
            
            # Check if x can be added
            # New triple sums involving x:
            # 3x
            # 2x + s for s in selected
            # x + 2s for s in selected  
            # x + s + t for s,t in selected, s<=t
            
            valid = True
            new_sums = set()
            
            # 3x
            s3 = 3 * x
            if s3 in selected_sums:
                valid = False
            else:
                new_sums.add(s3)
            
            if valid and selected:
                sel_arr = np.array(selected)
                
                # 2x + s for each s in selected
                sums_2x_s = 2 * x + sel_arr
                for sv in sums_2x_s:
                    sv = int(sv)
                    if sv in selected_sums:
                        valid = False
                        break
                    new_sums.add(sv)
            
            if valid and selected:
                # x + 2s for each s in selected
                sums_x_2s = x + 2 * sel_arr
                for sv in sums_x_2s:
                    sv = int(sv)
                    if sv in selected_sums:
                        valid = False
                        break
                    new_sums.add(sv)
            
            if valid and len(selected) >= 2:
                # x + s + t for s,t in selected, s<=t
                sel_list = selected
                for i in range(len(sel_list)):
                    for j in range(i, len(sel_list)):
                        sv = x + sel_list[i] + sel_list[j]
                        if sv in selected_sums:
                            valid = False
                            break
                        new_sums.add(sv)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                selected_sums.update(new_sums)
                if len(selected) >= len(best):
                    if len(selected) > len(best):
                        best = list(selected)
            
            current += 1
        
        # After greedy pass, try to fill in gaps
        # Try adding smaller values that were skipped
        if time.time() > deadline - 5:
            break
        
        # Try to add any value from 1 to n that wasn't tried
        for x in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if x in selected:
                continue
            
            # Check if x can be added
            valid = True
            new_sums = set()
            
            s3 = 3 * x
            if s3 in selected_sums:
                valid = False
            else:
                new_sums.add(s3)
            
            if valid and selected:
                sel_arr = np.array(selected)
                
                sums_2x_s = 2 * x + sel_arr
                for sv in sums_2x_s:
                    sv = int(sv)
                    if sv in selected_sums:
                        valid = False
                        break
                    new_sums.add(sv)
            
            if valid and selected:
                sums_x_2s = x + 2 * sel_arr
                for sv in sums_x_2s:
                    sv = int(sv)
                    if sv in selected_sums:
                        valid = False
                        break
                    new_sums.add(sv)
            
            if valid and len(selected) >= 2:
                sel_list = selected
                for i in range(len(sel_list)):
                    for j in range(i, len(sel_list)):
                        sv = x + sel_list[i] + sel_list[j]
                        if sv in selected_sums:
                            valid = False
                            break
                        new_sums.add(sv)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                selected_sums.update(new_sums)
                if len(selected) > len(best):
                    best = list(selected)
        
        if len(selected) > len(best):
            best = list(selected)
    
    if time.time() > deadline - 3:
        return best[:80] if len(best) >= 80 else best
    
    # Final verification and polish
    if len(best) > 0:
        # Verify the best set
        sel = sorted(best)
        all_sums = set()
        ok = True
        for i in range(len(sel)):
            for j in range(i, len(sel)):
                for k in range(j, len(sel)):
                    s = sel[i] + sel[j] + sel[k]
                    if s in all_sums:
                        ok = False
                        break
                    all_sums.add(s)
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Need to fix - remove conflicting elements
            # Just use a simple greedy rebuild
            sel2 = []
            sums2 = set()
            for x in sel:
                valid = True
                # Check all new triple sums
                for i in range(len(sel2)):
                    for j in range(i, len(sel2)):
                        for k in [j]:  # this is incomplete, just do simple check
                            pass
                # Simple check: just try adding
                test = sel2 + [x]
                ts = set()
                for i2 in range(len(test)):
                    for j2 in range(i2, len(test)):
                        for k2 in range(j2, len(test)):
                            s2 = test[i2] + test[j2] + test[k2]
                            if s2 in ts:
                                valid = False
                                break
                            ts.add(s2)
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    sel2.append(x)
                    sums2 = ts
            best = sel2
    
    return best[:80] if len(best) >= 80 else best

import random
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import defaultdict

def _check_add(x, selected, selected_sums):
    """Check if x can be added to selected set without triple sum collisions.
    Returns (valid, new_sums) where new_sums is the set of new triple sums."""
    new_sums = set()
    
    # 3x
    s3 = 3 * x
    if s3 in selected_sums:
        return False, set()
    new_sums.add(s3)
    
    if not selected:
        return True, new_sums
    
    sel_arr = np.array(selected)
    
    # 2x + s for each s in selected
    sums_2x_s = 2 * x + sel_arr
    for sv in sums_2x_s:
        sv = int(sv)
        if sv in selected_sums:
            return False, set()
        new_sums.add(sv)
    
    # x + 2s for each s in selected
    sums_x_2s = x + 2 * sel_arr
    for sv in sums_x_2s:
        sv = int(sv)
        if sv in selected_sums:
            return False, set()
        new_sums.add(sv)
    
    # x + s + t for s,t in selected, s<=t
    if len(selected) >= 2:
        sel_list = selected
        for i in range(len(sel_list)):
            for j in range(i, len(sel_list)):
                sv = x + sel_list[i] + sel_list[j]
                if sv in selected_sums:
                    return False, set()
                new_sums.add(sv)
    
    return True, new_sums

def _greedy_build(start_val=1, n=4096, deadline=None):
    """Greedy build from start_val upward, then try to fill gaps."""
    selected = []
    selected_sums = set()
    
    # Phase 1: Greedy from start_val upward
    current = start_val
    while current <= n:
        if deadline and time.time() > deadline:
            break
        
        valid, new_sums = _check_add(current, selected, selected_sums)
        if valid:
            selected.append(current)
            selected_sums.update(new_sums)
        
        current += 1
    
    # Phase 2: Try to fill in gaps - try adding values not yet in set
    for x in range(1, n + 1):
        if deadline and time.time() > deadline:
            break
        if x in selected:
            continue
        
        valid, new_sums = _check_add(x, selected, selected_sums)
        if valid:
            selected.append(x)
            selected_sums.update(new_sums)
    
    return selected, selected_sums

def _greedy_with_backtracking(n=4096, deadline=None, max_depth=15, branch_factor=3):
    """Greedy with limited backtracking when stuck."""
    selected = []
    selected_sums = set()
    current = 1
    best_size = 0
    
    def backtrack(start_from, depth):
        nonlocal best_size, selected, selected_sums
        
        if deadline and time.time() > deadline:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
        
        if depth <= 0:
            return
        
        # Try to continue greedily from start_from
        x = start_from
        found_valid = False
        while x <= n:
            if deadline and time.time() > deadline:
                return
            
            valid, new_sums = _check_add(x, selected, selected_sums)
            if valid:
                # Add x and continue
                selected.append(x)
                selected_sums.update(new_sums)
                found_valid = True
                
                if len(selected) > best_size:
                    best_size = len(selected)
                
                # Recurse to continue from x+1
                backtrack(x + 1, depth - 1)
                
                # Undo
                selected_sums -= new_sums
                selected.pop()
                
                # Try next candidate
                x += 1
            else:
                x += 1
        
        # If we couldn't add anything, try backtracking more aggressively
        # by removing the last element and trying alternatives
        if not found_valid and len(selected) > 0:
            last = selected[-1]
            # Try removing last and trying different values
            # This is simplified - just try a few alternatives
            for alt in range(last - branch_factor, last + branch_factor + 1):
                if alt <= 0 or alt > n or alt == last:
                    continue
                if alt in selected:
                    continue
                
                # Remove last element
                # We need to remove all sums involving last
                # This is complex, so for now just skip this optimization
                break
    
    backtrack(1, max_depth)
    
    return selected, selected_sums, best_size

def _local_search(initial_set, n=4096, deadline=None, max_iterations=100):
    """Try to improve a set by swapping elements."""
    selected = list(initial_set)
    selected_sums = set()
    
    # Rebuild sums
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            for k in range(j, len(selected)):
                s = selected[i] + selected[j] + selected[k]
                selected_sums.add(s)
    
    best = list(selected)
    best_size = len(selected)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Try removing one element and adding a better one
        for i in range(len(selected)):
            if deadline and time.time() > deadline:
                break
            
            removed = selected[i]
            
            # Try to find a replacement
            for x in range(1, n + 1):
                if deadline and time.time() > deadline:
                    break
                if x in selected or x == removed:
                    continue
                
                # Check if x can replace removed
                # This is complex - skip for now
                break
        
        # Try adding a new element
        for x in range(1, n + 1):
            if deadline and time.time() > deadline:
                break
            if x in selected:
                continue
            
            valid, new_sums = _check_add(x, selected, selected_sums)
            if valid:
                selected.append(x)
                selected_sums.update(new_sums)
                improved = True
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best = list(selected)
                
                break  # Only add one at a time
        
        if not improved:
            break
    
    return best, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    best = []
    best_size = 0
    
    # Strategy 1: Simple greedy from 1
    if time.time() < deadline - 10:
        selected, selected_sums = _greedy_build(1, n, deadline - 5)
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(selected)
    
    # Strategy 2: Greedy from different starting points
    for start_val in [2, 3, 5, 10, 20, 50, 100]:
        if time.time() > deadline - 5:
            break
        
        selected, selected_sums = _greedy_build(start_val, n, deadline - 5)
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(selected)
    
    # Strategy 3: Randomized greedy attempts
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation: start from a random point
        start_val = random.randint(1, 200)
        selected, selected_sums = _greedy_build(start_val, n, deadline - 5)
        if len(selected) > best_size:
            best = list(selected)
            best_size = len(selected)
    
    # Strategy 4: Local search to improve best
    if time.time() < deadline - 10 and best:
        improved, improved_size = _local_search(best, n, deadline - 5)
        if improved_size > best_size:
            best = improved
            best_size = improved_size
    
    # Strategy 5: Try greedy with backtracking on a subset
    if time.time() < deadline - 15:
        # Use backtracking on a smaller range first
        bt_selected, bt_sums, bt_best = _greedy_with_backtracking(
            min(n, 500), deadline - 5, max_depth=8, branch_factor=2
        )
        if bt_best > best_size:
            # Extend the backtracking result with greedy
            selected = list(bt_selected)
            selected_sums = set(bt_sums)
            
            # Continue greedily
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if x in selected:
                    continue
                
                valid, new_sums = _check_add(x, selected, selected_sums)
                if valid:
                    selected.append(x)
                    selected_sums.update(new_sums)
            
            if len(selected) > best_size:
                best = list(selected)
                best_size = len(selected)
    
    # Final verification
    if best and time.time() < deadline - 3:
        sel = sorted(best)
        all_sums = set()
        ok = True
        for i in range(len(sel)):
            for j in range(i, len(sel)):
                for k in range(j, len(sel)):
                    s = sel[i] + sel[j] + sel[k]
                    if s in all_sums:
                        ok = False
                        break
                    all_sums.add(s)
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Rebuild with simple greedy
            selected = []
            selected_sums = set()
            for x in sorted(best):
                valid, new_sums = _check_add(x, selected, selected_sums)
                if valid:
                    selected.append(x)
                    selected_sums.update(new_sums)
            best = selected
    
    return best[:80] if len(best) >= 80 else best

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
