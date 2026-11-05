import random
import time
import math

def construct():
    start_time = time.time()
    time_limit = 115
    N = 1000
    LAMBDA = 100
    
    def greedy_init():
        s = set()
        counts = {}
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            can_add = True
            for y in s:
                if counts.get(x + y, 0) >= 2:
                    can_add = False
                    break
            if can_add and counts.get(2 * x, 0) >= 2:
                can_add = False
            
            if can_add:
                s.add(x)
                for y in s:
                    sm = x + y
                    counts[sm] = counts.get(sm, 0) + 1
        return s, counts

    best_size = 0
    best_set = set()
    
    while time.time() - start_time < time_limit:
        s, counts = greedy_init()
        # Calculate initial violations
        total_violations = 0
        for c in counts.values():
            if c > 2:
                total_violations += c - 2
        
        current_score = len(s) - LAMBDA * total_violations
        
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.95
        
        while T > T_min and time.time() - start_time < time_limit:
            x = random.randint(1, N)
            
            if x in s:
                # Remove x
                # Delta size: -1
                # Delta violations:
                # For each sum s' involving x, count decreases by 1.
                # If count was 3, becomes 2 -> violations drop by 1.
                # If count was 2, becomes 1 -> violations drop by 0.
                # If count was 1, becomes 0 -> violations drop by 0.
                
                old_viol = total_violations
                s.remove(x)
                for y in s:
                    sm = x + y
                    old_c = counts[sm]
                    counts[sm] = old_c - 1
                    if old_c == 3:
                        total_violations -= 1
                    # if old_c > 3, violations drop by 1 per step? 
                    # max(0, c-2). 
                    # c=3 -> 1. c-1=2 -> 0. Drop 1.
                    # c=4 -> 2. c-1=3 -> 1. Drop 1.
                    # So if old_c > 2, total_violations -= 1.
                    # Wait, my logic above:
                    # if old_c == 3: drop 1.
                    # if old_c == 4: drop 1.
                    # So: if old_c > 2: total_violations -= 1.
                    
                # Handle 2x
                sm2 = 2 * x
                if sm2 in counts:
                    old_c = counts[sm2]
                    counts[sm2] = old_c - 1
                    if old_c > 2:
                        total_violations -= 1
                
                delta_viol = total_violations - old_viol
                delta_score = -1 - LAMBDA * delta_viol
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    # Accept
                    pass
                else:
                    # Reject: Add x back
                    s.add(x)
                    for y in s:
                        sm = x + y
                        new_c = counts[sm]
                        counts[sm] = new_c + 1
                        if new_c == 2: # Becomes 3, so violation increases by 1
                            total_violations += 1
                    sm2 = 2 * x
                    if sm2 in counts:
                        new_c = counts[sm2]
                        counts[sm2] = new_c + 1
                        if new_c == 2:
                            total_violations += 1
                    # Note: if new_c was 3, becomes 4. Violation increases by 1.
                    # So if new_c >= 2: total_violations += 1.
                    
                    # Correction:
                    # When adding back:
                    # old_c (in rejected state) -> new_c (in original state)
                    # If new_c > 2 and old_c <= 2? No.
                    # In rejected state, count was C. In original, count was C+1.
                    # Violation contribution: max(0, C-2) vs max(0, C+1-2).
                    # If C+1 > 2 and C <= 2, i.e., C=2, then 0 -> 1. Increase 1.
                    # If C+1 > 2 and C > 2, i.e., C>=3, then (C-2) -> (C-1). Increase 1.
                    # So if original count (new_c) > 2, violation increases by 1 when restoring.
                    
                    # My code above:
                    # if new_c == 2: total_violations += 1.
                    # This is WRONG for C=3 case (new_c=4).
                    # Should be: if new_c > 2: total_violations += 1.
                    
                    # Let's re-verify.
                    # Remove:
                    # Count C -> C-1.
                    # Viol: max(0, C-2) -> max(0, C-3).
                    # Diff: max(0, C-2) - max(0, C-3).
                    # If C=3: 1 - 0 = 1. (Decrease 1)
                    # If C=4: 2 - 1 = 1. (Decrease 1)
                    # If C=2: 0 - 0 = 0.
                    # So if C > 2, total_violations -= 1.
                    
                    # Add (Reject Remove):
                    # Count C-1 -> C.
                    # Viol: max(0, C-3) -> max(0, C-2).
                    # Diff: max(0, C-2) - max(0, C-3).
                    # If C=3: 1 - 0 = 1. (Increase 1)
                    # If C=4: 2 - 1 = 1. (Increase 1)
                    # If C=2: 0 - 0 = 0.
                    # So if C > 2, total_violations += 1.
                    
                    # So in Reject Remove block:
                    # We are restoring counts from C-1 to C.
                    # We check the NEW count (which is C).
                    # if new_c > 2: total_violations += 1.
                    
            else:
                # Add x
                old_viol = total_violations
                s.add(x)
                for y in s:
                    sm = x + y
                    old_c = counts.get(sm, 0)
                    counts[sm] = old_c + 1
                    if old_c >= 2: # Becomes >= 3
                        total_violations += 1
                sm2 = 2 * x
                old_c = counts.get(sm2, 0)
                counts[sm2] = old_c + 1
                if old_c >= 2:
                    total_violations += 1
                
                delta_viol = total_violations - old_viol
                delta_score = 1 - LAMBDA * delta_viol
                
                if delta_score > 0 or random.random() < math.exp(delta_score / T):
                    # Accept
                    pass
                else:
                    # Reject: Remove x
                    s.remove(x)
                    for y in s:
                        sm = x + y
                        new_c = counts[sm] # This is the count AFTER removal (old_c)
                        counts[sm] = new_c - 1
                        if new_c > 2: # Original count was new_c, which was > 2
                            total_violations -= 1
                    sm2 = 2 * x
                    new_c = counts[sm2]
                    counts[sm2] = new_c - 1
                    if new_c > 2:
                        total_violations -= 1
                    
            T *= cooling_rate
        
        # End SA
        if total_violations == 0:
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)
        else:
            # Repair
            # Simple repair: remove elements causing violations
            # We can just discard this solution if it's too bad, 
            # or try to repair.
            # Given time constraints, maybe just discard if not valid?
            # But greedy init usually produces valid sets.
            # SA might break validity.
            # Let's try a quick repair.
            s_rep = set(s)
            counts_rep = dict(counts)
            changed = True
            while changed and time.time() - start_time < time_limit - 2:
                changed = False
                for sm, c in list(counts_rep.items()):
                    if c > 2:
                        # Find pair
                        for a in s_rep:
                            b = sm - a
                            if b in s_rep and a <= b:
                                # Remove a
                                s_rep.remove(a)
                                for y in s_rep:
                                    sm2 = a + y
                                    if sm2 in counts_rep:
                                        counts_rep[sm2] -= 1
                                sm3 = 2 * a
                                if sm3 in counts_rep:
                                    counts_rep[sm3] -= 1
                                changed = True
                                break
                        if changed:
                            break
            if len(s_rep) > best_size:
                # Verify validity
                valid = True
                for c in counts_rep.values():
                    if c > 2:
                        valid = False
                        break
                if valid:
                    best_size = len(s_rep)
                    best_set = set(s_rep)
                    
        if time.time() - start_time > time_limit - 5:
            break

    return list(best_set)
