import random

N = 256

def construct():
    # Use a deterministic seed for reproducibility if needed, 
    # but the problem doesn't require it. 
    # We'll stick to deterministic greedy for stability.
    
    S = []
    Sigma = set()
    
    # blocked[z] = True if z is blocked by current S
    # Initially, no sums, so nothing is blocked.
    # But we need to track which elements are "available" to be added.
    # An element x is a candidate if it is NOT in S and is compatible with S.
    # We want to pick x that maximizes the number of OTHER elements it blocks.
    
    # To speed up, we maintain a set of "unblocked" elements.
    # Initially all are unblocked.
    unblocked = set(range(1, N + 1))
    
    while unblocked:
        best_x = None
        best_gain = -1
        
        # Iterate through candidates in unblocked
        # We can shuffle to break ties, or just pick first max.
        # Deterministic: pick smallest x with max gain? Or just max gain.
        # Let's iterate all unblocked.
        
        for x in unblocked:
            # Check if x is compatible with S (Sidon)
            # New sums: x+s for s in S, and 2x
            new_sums = [x + s for s in S] + [2 * x]
            
            # Check distinctness within new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
                
            # Check intersection with existing Sigma
            if Sigma.intersection(new_sums):
                continue
                
            # If compatible, calculate gain
            # Gain = number of elements z in unblocked \ {x} that become blocked
            # by adding x to S and new_sums to Sigma.
            
            # Simulate new state
            # New S = S + [x]
            # New Sigma = Sigma U new_sums
            
            # An element z is blocked if:
            # 1. 2z is in New Sigma
            # 2. Exists s in New S such that z+s is in New Sigma
            
            gain = 0
            for z in unblocked:
                if z == x:
                    continue
                # Check if z is blocked by New Sigma and New S
                blocked_z = False
                # Check diagonal
                if 2 * z in Sigma or 2 * z in new_sums:
                    blocked_z = True
                else:
                    # Check off-diagonal
                    # For each s in New S, check if z+s in New Sigma
                    # Note: z+s is the same as s+z.
                    # We just need to check if the sum exists.
                    for s in S:
                        if z + s in Sigma or z + s in new_sums:
                            blocked_z = True
                            break
                    if not blocked_z:
                        # Check with s=x
                        if z + x in new_sums: # z+x is in new_sums? 
                           # new_sums contains x+s for s in S. 
                           # Does it contain z+x? Only if z in S, but z not in S.
                           # Wait, new_sums = {x+s | s in S} U {2x}.
                           # z+x is not in new_sums unless z in S (false) or 2x = z+x => z=x (false).
                           # So z+x is not in new_sums.
                           # But z+x might be in Sigma? No, if it was, z would have been blocked already.
                           # So we only need to check if z+s (s in S) is in New Sigma.
                           # We already checked z+s in Sigma in the previous loop?
                           # No, we checked `z+s in Sigma or z+s in new_sums`.
                           # If z+s in Sigma, z was already blocked.
                           # So if z is in `unblocked`, then z+s is NOT in Sigma for any s in S.
                           # So we only need to check if z+s is in new_sums.
                           pass
            
            # The logic above is getting messy. 
            # Simpler: Just check if z is blocked by (S U {x}, Sigma U new_sums).
            
            # Let's refactor the check.
            
        # This inner loop is O(|U|^2 * |S|). 
        # |U| up to 256. 256^2 * 10 = 650k. 
        # Times k (10) = 6.5M. OK.
        
        # Let's implement the gain calculation carefully.
        
        for x in unblocked:
            # Sidon check
            new_sums = [x + s for s in S] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if Sigma.intersection(new_sums):
                continue
            
            # Calculate gain
            gain = 0
            for z in unblocked:
                if z == x:
                    continue
                # Is z blocked by S' = S U {x} and Sigma' = Sigma U new_sums?
                is_blocked = False
                # 1. Diagonal: 2z in Sigma'
                if 2 * z in Sigma or 2 * z in new_sums:
                    is_blocked = True
                else:
                    # 2. Off-diagonal: exists s in S' such that z+s in Sigma'
                    # s can be from S or x.
                    # If s in S: z+s in Sigma' means z+s in Sigma OR z+s in new_sums.
                    # Since z is currently unblocked, z+s is NOT in Sigma for any s in S.
                    # So we only need to check if z+s is in new_sums.
                    # If s = x: z+x in Sigma' means z+x in Sigma OR z+x in new_sums.
                    # z+x is not in Sigma (else z blocked).
                    # Is z+x in new_sums?
                    # new_sums = {x+s_i} U {2x}.
                    # z+x = x+s_i => z = s_i. But z not in S. So no.
                    # z+x = 2x => z=x. But z!=x. So no.
                    # So z+x is never in new_sums.
                    
                    # So we only need to check:
                    # Does there exist s in S such that z+s is in new_sums?
                    
                    for s in S:
                        if z + s in new_sums:
                            is_blocked = True
                            break
                    
                if is_blocked:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # Should not happen if unblocked is not empty and we can always add something?
            # Actually, if S is empty, we can add anything.
            # If S is not empty, maybe no x in unblocked is compatible?
            # Then S is maximal already?
            # If no x in unblocked is compatible, then all x in unblocked are "blocked" 
            # in the sense that they cannot be added without violating Sidon.
            # But maximality requires that they are blocked by *sum collisions*.
            # Incompatibility IS a sum collision.
            # So if no x is compatible, S is maximal.
            break
        
        # Add best_x to S
        S.append(best_x)
        Sigma.update([best_x + s for s in S[:-1]] + [2 * best_x])
        unblocked.remove(best_x)
        
        # Update unblocked: remove any z that is now blocked
        # We can just recalculate unblocked from scratch or update.
        # Recalculation is O(N * |S|).
        new_unblocked = set()
        for z in unblocked:
            # Check if z is blocked by current S and Sigma
            blocked_z = False
            if 2 * z in Sigma:
                blocked_z = True
            else:
                for s in S:
                    if z + s in Sigma:
                        blocked_z = True
                        break
            if not blocked_z:
                new_unblocked.add(z)
        unblocked = new_unblocked
        
    return S
