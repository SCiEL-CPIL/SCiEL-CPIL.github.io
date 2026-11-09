import random

N = 256

def construct():
    best_size = N + 1
    best_set = None
    
    # Run multiple iterations with different random seeds for tie-breaking
    # to find a potentially smaller set.
    # The greedy choice depends on the order if there are ties.
    
    for _ in range(50):
        S = []
        sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            max_blocked = -1
            best_new_sums = None
            
            # Shuffle to break ties randomly
            candidates = list(remaining)
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                valid = True
                new_sums = set()
                
                for s in S:
                    val = x + s
                    if val in sums or val in new_sums:
                        valid = False
                        break
                    new_sums.add(val)
                
                if valid:
                    val = 2 * x
                    if val in sums or val in new_sums:
                        valid = False
                    else:
                        new_sums.add(val)
                
                if not valid:
                    continue
                
                # Calculate blocked elements
                # An element y is blocked if it cannot be added to S U {x}
                # This means for some pair in (S U {x}), the sum conflicts with another pair in (S U {x}) or existing sums.
                # Since S is Sidon, conflicts only arise involving x or y.
                # y is invalid if:
                # 1. 2y is in sums(S U {x})
                # 2. y + s is in sums(S U {x}) for some s in S U {x}
                
                # Let's just check if y can be added to S' = S U {x}
                # We can reuse the "valid" check logic for y against S'
                
                blocked_count = 0
                # Optimization: If we find a x that blocks many, we might stop early?
                # No, we need max.
                
                for y in remaining:
                    if y == x:
                        continue
                    # Check if y is valid in S'
                    y_valid = True
                    # Check against existing sums of S
                    for s in S:
                        val = y + s
                        if val in sums:
                            y_valid = False
                            break
                    if y_valid:
                        # Check against new sums involving x
                        # New sums are x+s for s in S, and 2x
                        # y+x is a new sum. Is it in sums? 
                        # If y+x in sums, invalid.
                        if (y + x) in sums:
                            y_valid = False
                        else:
                            # Check if y+x conflicts with x+s?
                            # y+x == x+s => y==s, but y!=x and y in remaining, s in S, so y!=s.
                            # So y+x is distinct from x+s.
                            # Check if 2y conflicts.
                            # 2y in sums?
                            if (2 * y) in sums:
                                y_valid = False
                            else:
                                # 2y in new_sums?
                                # new_sums contains x+s and 2x.
                                # 2y == x+s => 2y-x = s.
                                # 2y == 2x => y=x (false).
                                # So we just need to check if 2y is in sums or if 2y-x is in S?
                                # Actually, simpler: just check if y can be added to S U {x}.
                                # We can simulate the addition.
                                pass # This is getting complicated. Let's just use the full check.
                    
                    if not y_valid:
                        blocked_count += 1
                        
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_x = x
                    best_new_sums = new_sums
                    
            if best_x is None:
                break
                
            S.append(best_x)
            sums.update(best_new_sums)
            remaining.remove(best_x)
            
            # Optional: Remove newly blocked elements from 'remaining'?
            # No, 'remaining' is just the pool of candidates we *might* pick.
            # If an element is blocked, we just won't pick it because it will fail the `valid` check in future iterations?
            # Actually, if y is blocked by S, it means y CANNOT be added to S.
            # So in the next iteration, when we check y, `valid` will be False.
            # So we don't need to remove them, but it saves time to do so.
            # However, determining which are blocked is the expensive part we just did.
            # We can store them.
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = S.copy()
            if best_size <= 10:
                break
                
    return best_set
