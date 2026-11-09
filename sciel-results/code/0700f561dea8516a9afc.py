import random
import time

N = 256

def construct():
    best_set = None
    best_size = N + 1
    start_time = time.time()
    time_limit = 25.0  # Leave buffer for overhead
    
    def greedy_max_coverage():
        selected = []
        sums = set()
        available = list(range(1, N + 1))
        
        while available:
            best_x = None
            max_blocks = -1
            
            # Randomize to break ties and explore different solutions
            # We iterate through a shuffled copy to pick the best x
            # To save time, we might not need to check ALL x if we find a good one early?
            # No, we need max blocks.
            
            # Optimization: If we just want *a* good set, we can randomize the tie-breaking
            # by shuffling available and picking the first one that has high blocks?
            # No, let's find the true max for this step.
            
            # To speed up, we can pre-compute some things?
            # For now, direct calculation.
            
            # We need to check compatibility of x with S.
            # If x is not compatible, it's not a candidate.
            
            candidates = []
            for x in available:
                # Check if x is compatible with S
                valid = True
                # Check 2x
                if 2 * x in sums:
                    valid = False
                else:
                    for s in selected:
                        if (x + s) in sums:
                            valid = False
                            break
                if valid:
                    candidates.append(x)
                    
            if not candidates:
                break
                
            # Now, for each candidate x, count blocks
            # To make it faster, we can shuffle candidates and pick the best?
            # Or just iterate all.
            
            best_x = None
            max_blocks = -1
            # Shuffle candidates to handle ties randomly
            random.shuffle(candidates)
            
            for x in candidates:
                # Calculate blocks for x
                # S_new = S U {x}
                # Sums_new = sums U {x+s for s in S} U {2x}
                
                blocks = 0
                # Check each y in available (excluding x)
                # We can optimize by only checking y that are compatible with S?
                # If y is not compatible with S, it's already "dead".
                # But we maintain `available` as compatible with S.
                # So all y in available are compatible with S.
                
                # When we add x, y becomes incompatible if:
                # 1. 2y is in Sums_new
                # 2. y + s is in Sums_new for some s in S_new (s != y)
                
                # Sums_new contains:
                # - sums (from S)
                # - x + s (for s in S)
                # - 2x
                
                # Check condition 1: 2y in Sums_new
                # 2y in sums? No, y compatible with S implies 2y not in sums.
                # 2y == x + s?
                # 2y == 2x? -> y=x (skipped)
                
                # Check condition 2: y + s in Sums_new
                # s can be x or s in S.
                # Case A: s = x. y + x in Sums_new?
                # y + x in sums? (No, y,x compatible with S individually, but y+x might be in sums? 
                # If y+x in sums, then y is incompatible with S? 
                # Wait. If y+x = a+b for a,b in S, then y is NOT compatible with S?
                # No. Compatibility of y with S means:
                # 2y not in sums, and y+s not in sums for all s in S.
                # It does NOT check y+x because x is not in S yet.
                # So y+x CAN be in sums.
                # If y+x is in sums, then adding x makes y incompatible because y+x is a new sum for y (with x) and it collides with existing sum.
                
                # So, for each y:
                # It is blocked if:
                # - 2y == 2x (impossible)
                # - 2y == x + s for some s in S
                # - y + x == 2x (impossible)
                # - y + x == s + t for some s,t in S (i.e., y+x in sums)
                # - y + x == x + s (implies y=s, but y!=x and y,s in S? No, y not in S)
                #   Wait, y+x is a sum involving x.
                #   Collision if y+x equals any sum in Sums_new.
                #   Sums_new = sums U {x+s} U {2x}.
                #   So y+x in sums? OR y+x in {x+s} (implies y=s, impossible) OR y+x=2x (implies y=x, impossible).
                #   So y+x in sums is the only check for the x+s part?
                #   Wait, y+x is a sum of y and x.
                #   The existing sums in Sums_new are from S and from x with S.
                #   If y+x = a+b (a,b in S), collision.
                #   If y+x = x+s (s in S), collision -> y=s. But y not in S.
                #   If y+x = 2x, collision -> y=x.
                
                # Also check y + s (s in S) in Sums_new.
                # y+s in sums? (No, y compatible with S)
                # y+s == x+t (t in S)?
                # y+s == 2x?
                
                # This is getting complex to optimize manually.
                # Let's just compute Sums_new as a set and check membership.
                # Sums_new size is |S| + 1 + |S| = 2|S|+1 ~ 21.
                # Membership check is O(1).
                
                # Construct Sums_new temporarily?
                # Or just check conditions.
                
                # Let's construct the set of new sums for x: new_sums_x
                new_sums_x = [2*x]
                for s in selected:
                    new_sums_x.append(x+s)
                
                # For each y in available:
                # y is blocked if:
                # 1. 2y is in sums OR 2y is in new_sums_x
                # 2. (y+s) is in sums OR (y+s) is in new_sums_x for ANY s in selected OR s=x
                
                # Note: if y+s is in sums, y was incompatible with S. But we filtered candidates?
                # No, `available` contains y compatible with S.
                # So y+s NOT in sums.
                # So we only need to check if y+s is in new_sums_x.
                
                # Similarly, 2y not in sums.
                # So check if 2y in new_sums_x.
                
                # And check if y+x is in sums (from condition y+x in Sums_new via 'sums' part).
                # And check if y+x is in new_sums_x? 
                # y+x in new_sums_x means y+x = 2x (y=x, no) or y+x = x+s (y=s, no).
                # So y+x is NOT in new_sums_x (for y!=x, s!=y).
                # Wait, new_sums_x contains x+s.
                # If y+x = x+s, then y=s.
                
                # So for y to be blocked:
                # A. 2y in new_sums_x
                # B. y+x in sums
                # C. (y+s) in new_sums_x for some s in selected
                
                # Let's verify C.
                # y+s is a sum involving y and s (from S).
                # In S_new, this sum exists.
                # Collision if it equals an existing sum in Sums_new.
                # Existing sums in Sums_new: sums (from S) and new_sums_x (from x).
                # y+s in sums? No (y compatible with S).
                # y+s in new_sums_x?
                # new_sums_x contains 2x and x+t.
                # So y+s = 2x or y+s = x+t.
                
                # So conditions for y being blocked by x:
                # 1. 2y == 2x (No)
                # 2. 2y == x+t for some t in S
                # 3. y+x in sums
                # 4. y+s == 2x for some s in S
                # 5. y+s == x+t for some s,t in S
                
                # We can precompute some sets for S to speed this up?
                # For now, with |S|~10, checking these conditions is fast.
                
                is_blocked = False
                # Check 2y in new_sums_x
                if 2*y in new_sums_x:
                    is_blocked = True
                elif (y+x) in sums:
                    is_blocked = True
                else:
                    # Check y+s in new_sums_x
                    # We can just iterate s in selected
                    for s in selected:
                        if (y+s) in new_sums_x:
                            is_blocked = True
                            break
                            
                if is_blocked:
                    blocks += 1
                    
                if blocks > max_blocks:
                    max_blocks = blocks
                    best_x = x
                    # Early exit if blocks is very high? No, we want max.
            
            if best_x is None:
                break
                
            selected.append(best_x)
            # Update sums
            for s in selected[:-1]:
                sums.add(best_x + s)
            sums.add(2 * best_x)
            
            # Remove best_x from available
            available.remove(best_x)
            
            # Prune available
            new_available = []
            for y in available:
                # Check if y compatible with new S
                # 2y in sums?
                if 2*y in sums:
                    continue
                compat = True
                for s in selected:
                    if (y+s) in sums:
                        compat = False
                        break
                if compat:
                    new_available.append(y)
            available = new_available

        return selected

    iterations = 0
    while time.time() - start_time < time_limit:
        s = greedy_max_coverage()
        if len(s) < best_size:
            best_size = len(s)
            best_set = s.copy()
            if best_size <= 9:
                break
        iterations += 1

    return best_set
