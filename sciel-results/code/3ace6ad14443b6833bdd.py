import random
import time
import math

def construct():
    # We want to minimize n.
    # Try n from 31 upwards.
    # If we fail, fall back to Paley 83.
    
    def check_and_build(n, time_limit=10.0):
        """
        Attempts to find a valid tournament on n vertices using simulated annealing.
        Returns orientation list if found, else None.
        """
        if n < 4:
            return None
            
        # Precompute all 4-sets
        sets4 = []
        # For efficient update, we can just iterate all 4-sets if n is small, 
        # but for n=31, 31k sets is fine to check on flip if we optimize.
        # However, checking ALL 31k sets on every flip is 31k * 4 ops = 120k ops.
        # 120k ops is ~0.01s in Python? Maybe 0.05s.
        # 120s / 0.05s = 2400 flips. Might be too few for SA.
        
        # Better: Only check affected sets.
        # Precompute for each vertex v, the list of 4-sets containing v.
        sets_containing = [[] for _ in range(n)]
        all_sets = []
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        s = (a, b, c, d)
                        all_sets.append(s)
                        sets_containing[a].append(s)
                        sets_containing[b].append(s)
                        sets_containing[c].append(s)
                        sets_containing[d].append(s)
                        
        # Initialize random tournament
        # InMask[i] is a bitmask of vertices pointing TO i
        InMask = [0] * n
        # adj[i][j] = 1 if i -> j? 
        # We just use InMask.
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    # i -> j
                    InMask[j] |= (1 << i)
                else:
                    # j -> i
                    InMask[i] |= (1 << j)
                    
        # Calculate initial bad count
        bad_count = 0
        # We can store bad status to avoid re-checking? 
        # Or just re-calculate for affected sets.
        # Let's store is_bad[s] to know if it was bad before.
        is_bad = [False] * len(all_sets)
        
        for idx, s in enumerate(all_sets):
            a, b, c, d = s
            if (InMask[a] & InMask[b] & InMask[c] & InMask[d]) == 0:
                is_bad[idx] = True
                bad_count += 1
                
        if bad_count == 0:
            return build_orientation(InMask, n)

        T = 1.0
        T_min = 0.01
        cooling = 0.999
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if bad_count == 0:
                return build_orientation(InMask, n)
                
            # Pick random edge to flip
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            # Determine current direction
            # If i -> j, bit i is in InMask[j]
            if InMask[j] & (1 << i):
                # i -> j
                # Flip to j -> i
                # Remove i from InMask[j]
                # Add j to InMask[i]
                old_dir = 0 # i->j
                new_dir = 1 # j->i
            else:
                # j -> i
                # Flip to i -> j
                # Remove j from InMask[i]
                # Add i to InMask[j]
                old_dir = 1
                new_dir = 0
                
            # Identify affected 4-sets
            # Union of sets containing i and sets containing j
            # We need to check if their bad status changes.
            # To avoid double counting, we can use a set of indices or just a boolean array visited?
            # Or simply iterate sets_containing[i] and sets_containing[j] and handle overlaps.
            # Overlap: sets containing both i and j.
            
            # Let's just iterate all affected and calculate new badness.
            # We need to know the OLD badness to calculate delta.
            # We have is_bad array.
            
            # To be efficient, we can collect unique indices.
            # Given n is small, sets_containing[i] has ~ (n-1 choose 3) ~ 4000 elements.
            # Union is ~ 8000.
            
            affected_indices = set()
            for idx in sets_containing[i]:
                affected_indices.add(idx)
            for idx in sets_containing[j]:
                affected_indices.add(idx)
                
            delta = 0
            # We will update InMask AFTER checking?
            # No, we need to check NEW state.
            # So we must apply the flip, check, and revert if rejected.
            
            # Apply flip
            if old_dir == 0: # i->j becomes j->i
                InMask[j] &= ~(1 << i)
                InMask[i] |= (1 << j)
            else: # j->i becomes i->j
                InMask[i] &= ~(1 << j)
                InMask[j] |= (1 << i)
                
            # Calculate new bad count for affected sets
            for idx in affected_indices:
                a, b, c, d = all_sets[idx]
                new_bad = ((InMask[a] & InMask[b] & InMask[c] & InMask[d]) == 0)
                if new_bad != is_bad[idx]:
                    if new_bad:
                        delta += 1
                    else:
                        delta -= 1
                    is_bad[idx] = new_bad
                    
            # Acceptance criterion
            if delta <= 0 or random.random() < math.exp(-delta / T):
                bad_count += delta
            else:
                # Revert
                if old_dir == 0:
                    InMask[j] |= (1 << i)
                    InMask[i] &= ~(1 << j)
                else:
                    InMask[i] |= (1 << j)
                    InMask[j] &= ~(1 << i)
                
                # Revert is_bad and bad_count
                # We updated is_bad in the loop above?
                # Yes, we updated is_bad[idx] = new_bad.
                # So we must revert is_bad too.
                for idx in affected_indices:
                    a, b, c, d = all_sets[idx]
                    # Recalculate OLD badness? 
                    # Or just flip back the is_bad flag if it changed?
                    # It's safer to just re-calculate or store old state.
                    # Let's just re-calculate old state for affected sets on revert.
                    # But we don't have old InMask.
                    # So we must have stored the old is_bad values?
                    # Or we can just accept the complexity of re-calculating.
                    # Actually, if we revert InMask, we can just re-evaluate badness for affected sets.
                    pass
                
                # Simpler revert:
                # We changed is_bad[idx] to new_bad.
                # We need to set it back to old_bad.
                # We didn't store old_bad.
                # Let's modify the loop to store changes.
                
                # To keep it simple and robust, let's just re-calculate bad_count for affected sets
                # after reverting InMask.
                # And reset is_bad for affected sets.
                
                # Wait, the loop above already updated is_bad.
                # So on revert, we must re-calculate is_bad for affected sets based on current (reverted) InMask.
                for idx in affected_indices:
                    a, b, c, d = all_sets[idx]
                    is_bad[idx] = ((InMask[a] & InMask[b] & InMask[c] & InMask[d]) == 0)
                
                # Recalculate bad_count contribution of affected sets
                # We need to subtract the delta we added? No, we didn't add it to bad_count yet.
                # We calculated delta. If rejected, we don't update bad_count.
                # But is_bad is now correct for the reverted state.
                # We just need to ensure bad_count is consistent.
                # bad_count was NOT updated.
                # So bad_count is still correct for the OLD state.
                # And is_bad is now correct for the OLD state (after re-calc).
                # So we are good.
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return None

    def build_orientation(InMask, n):
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                # If i -> j, bit i is set in InMask[j]
                if InMask[j] & (1 << i):
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Try increasing n
    for n in range(31, 84):
        # Use more time for smaller n
        t_limit = 10.0 if n < 40 else 5.0
        res = check_and_build(n, t_limit)
        if res is not None:
            return {'n': n, 'orientation': res}
            
    # Fallback to Paley 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
