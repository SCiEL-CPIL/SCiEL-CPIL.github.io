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
