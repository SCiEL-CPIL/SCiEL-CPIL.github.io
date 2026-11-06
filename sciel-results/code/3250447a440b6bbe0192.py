import random
import time

N = 256

def construct():
    best_size = N + 1
    best_set = None
    
    # Bitmask approach for speed
    # forbidden_mask: bit i is 1 if value i is forbidden (cannot be added)
    # We maintain a set of "valid" candidates implicitly by checking the mask
    
    # Precompute sums for quick lookup? 
    # Instead, we maintain a boolean array or bitmask for sums.
    # Max sum is 256+256=512.
    # We can use a set of sums or a bitmask of size 513.
    
    # Let's use a set of sums for simplicity, but optimize the candidate check.
    
    for trial in range(500):
        # Randomize the order of candidates to break symmetry
        # Or just use a fixed order but randomize the "tie-breaking" or selection heuristic?
        # The directive says: "prioritizing elements that maximize the number of newly blocked candidates"
        
        selected = []
        sums = set()
        # candidate_mask: 1 if candidate is available
        # Initially all 1..N are available
        # We'll maintain a list of available candidates to iterate over
        
        # To speed up "max blocked" calculation, we can precompute for each potential x,
        # which other y's it would block given the current state.
        # But the state changes.
        
        # Let's stick to the greedy but optimize the inner loop.
        # Key optimization: Instead of iterating all valid y's for each x,
        # we can precompute the "impact" of adding x.
        
        # Actually, a faster heuristic:
        # Pick x that maximizes the number of y in candidates such that y is blocked by x.
        # y is blocked by x if:
        # 1. y+x in sums (new sum) -> wait, y+x is the new sum.
        #    y is blocked if for some s in selected, y+s == x+s' (for some s' in selected U {x})?
        #    No. y is blocked if adding x creates a conflict for y.
        #    Conflict for y means:
        #    a) y+y is in sums (old or new)
        #    b) y+s is in sums (old or new) for some s in selected U {x}
        
        # Let's just run the greedy with the "max destruction" heuristic but optimize the Python loops.
        # We can use numpy? No, let's stick to pure Python but optimize.
        
        # Optimization 1: Use a list for candidates, remove from it.
        # Optimization 2: Early termination in inner loops.
        
        candidates = list(range(1, N + 1))
        # Shuffle to break symmetry?
        random.shuffle(candidates)
        
        while candidates:
            # Find valid candidates
            valid = []
            for x in candidates:
                ok = True
                # Check 2x
                if 2 * x in sums:
                    ok = False
                else:
                    # Check x+s
                    for s in selected:
                        if x + s in sums:
                            ok = False
                            break
                if ok:
                    valid.append(x)
            
            if not valid:
                break
            
            # Heuristic: Max destruction
            best_x = None
            max_invalidated = -1
            
            # Optimization: If valid is large, maybe sample?
            # But N=256 is small enough.
            # Let's try to optimize the "invalidated" count.
            
            # For each x in valid:
            #   new_sums = {x+s for s in selected} U {2x}
            #   invalidated = 0
            #   for y in valid:
            #     if y == x: continue
            #     if 2y in new_sums: invalidated++
            #     else:
            #       for s in selected:
            #         if y+s in new_sums:
            #           invalidated++
            #           break
            
            # This is O(V^2 * S).
            
            # Alternative: 
            # Precompute for each y in valid, what sums would block it?
            # y is blocked by new sum Z if Z == 2y or Z == y+s for some s in selected.
            # So for a given x, new_sums are known.
            # We can check if any new sum Z blocks y.
            
            # Let's just run it. 500 trials * 20 steps * 256^2 * 20 ops?
            # 500 * 20 * 65536 * 20 = 1.3e10. Too slow.
            
            # We need a faster heuristic.
            # "Max destruction" is expensive.
            # What if we pick the x that maximizes the number of *direct* conflicts?
            # Or just pick a random valid x?
            # The current program (score 1.1429) uses max destruction and gets ~17?
            # 1.0 is seed. 1.1429 means it improved by 14%.
            # If seed size is S, current size is S/1.1429? No, score = seed/value.
            # So value (size) = seed_size / score.
            # If seed_size is e.g. 20, current size is 17.5?
            
            # Let's try a different heuristic: 
            # Pick x that maximizes the number of *other* candidates that become invalid 
            # specifically due to the new sums involving x.
            
            # Actually, let's try to reduce the number of trials and make each trial faster.
            # Or use a smarter selection.
            
            # Heuristic: Pick the smallest valid x? (Fast)
            # Heuristic: Pick the largest valid x?
            # Heuristic: Pick x that is "central"?
            
            # Let's try: Pick x that maximizes the count of y in candidates 
            # such that y+x is in the range [1, 2N] and (y+x) would be a "bad" sum for y?
            # No.
            
            # Let's try: For each valid x, count how many y in candidates 
            # satisfy: (y+x in sums) OR (2y in {x+s for s in selected} U {2x})?
            # Wait, y is blocked if y+s' is in new_sums for some s' in selected U {x}.
            # new_sums = {x+s for s in selected} U {2x}.
            # So y is blocked if:
            # 1. 2y is in new_sums
            # 2. y+s is in new_sums for some s in selected
            
            # Let's precompute for each y in candidates, the set of sums that would block it.
            # Blockers(y) = {2y} U {y+s for s in selected}
            # y is valid if Blockers(y) is disjoint from sums.
            # When we add x, new_sums = {x+s for s in selected} U {2x}.
            # y becomes invalid if new_sums intersects Blockers(y).
            
            # So for each x, we want to maximize |{y in valid : new_sums(x) intersects Blockers(y)}|
            
            # We can precompute Blockers(y) for all y in candidates.
            # Then for each x, compute new_sums(x) and check intersection.
            
            # Cost: 
            # Precompute Blockers: O(N * S)
            # For each x in valid: O(S) to compute new_sums.
            # For each y in valid: O(1) to check intersection if we use sets? 
            # Intersection of two sets of size ~S. O(S).
            # Total: O(V * S * V) = O(V^2 * S). Same complexity.
            
            # But we can optimize the intersection check.
            # Blockers(y) is small (size S+1).
            # new_sums(x) is small (size S+1).
            
            # Let's just try to speed up the current max destruction heuristic 
            # by using lists and early breaks, and reduce trials to 100.
            
            # Or, use a different heuristic: 
            # Pick x that maximizes the number of *immediate* conflicts with existing sums?
            # No, x is valid, so it has no conflicts.
            
            # Let's try: Pick x that maximizes the number of y in candidates 
            # such that y+x is a "sum" that is already in sums? 
            # No, y+x is a new sum.
            
            # Let's try: Pick x that is "dense" in the sense that it creates many new sums 
            # that are close to other candidates' blockers.
            
            # Given time constraints, let's just optimize the existing loop.
            
            # Optimization: Use a local variable for sums, selected.
            # Use list for valid.
            
            for x in valid:
                # Compute new sums
                # new_sums = {x+s for s in selected} U {2x}
                # Check how many y in valid are blocked by these new sums.
                
                # Precomputed blockers for y?
                # Let's compute on the fly but efficiently.
                
                invalidated = 0
                # Create a set of new sums for fast lookup?
                # new_sums_set = {x+s for s in selected}
                # new_sums_set.add(2*x)
                
                # This is O(S).
                # Then for each y in valid (O(V)), check if 2y in new_sums_set or any y+s in new_sums_set.
                # The "any y+s" part is O(S).
                # Total O(V*S) per x. Total O(V^2*S).
                
                # Let's try to prune.
                # If invalidated is already less than current max, and remaining y's can't beat it?
                # Hard to bound.
                
                # Let's just run 100 trials with the current logic but optimized.
                pass
            
            # Actually, let's try a simpler heuristic that might be faster:
            # Pick the valid x that has the largest value? 
            # Or the smallest?
            # Random?
            
            # Let's try: Pick x that maximizes the number of y in candidates 
            # such that y+x <= N and y+x is in the "forbidden" zone?
            
            # Let's go with a hybrid:
            # 1. Compute valid list.
            # 2. If len(valid) < 10, use max destruction.
            # 3. Else, use a faster heuristic: 
            #    Pick x that maximizes the number of y in valid 
            #    such that y+x is in sums? (No, y+x is new)
            #    Pick x that maximizes the number of y in valid 
            #    such that 2y is in {x+s for s in selected} U {2x}?
            
            # Let's just use the max destruction heuristic but with a time limit per trial?
            # No, we need to finish in 30s.
            
            # Let's reduce trials to 50 and optimize the inner loop.
            
            best_x = None
            max_invalidated = -1
            
            for x in valid:
                # Quick check: does x block many?
                # Approximation: count how many y in valid have y+x in sums? 
                # No, y+x is new.
                
                # Let's just do the full check but break early if possible.
                invalidated = 0
                # Precompute new sums as a set
                new_sums = set()
                for s in selected:
                    new_sums.add(x + s)
                new_sums.add(2 * x)
                
                for y in valid:
                    if y == x:
                        continue
                    # Check if y is blocked by new_sums
                    blocked = False
                    if 2 * y in new_sums:
                        blocked = True
                    else:
                        for s in selected:
                            if y + s in new_sums:
                                blocked = True
                                break
                    if blocked:
                        invalidated += 1
                
                if invalidated > max_invalidated:
                    max_invalidated = invalidated
                    best_x = x
                    if max_invalidated == len(valid) - 1:
                        break # Can't do better
            
            # Commit best_x
            selected.append(best_x)
            for s in selected[:-1]:
                sums.add(best_x + s)
            sums.add(2 * best_x)
            
            # Update candidates
            # Remove best_x and any now-invalid candidates
            new_candidates = []
            for y in candidates:
                if y == best_x:
                    continue
                # Check if y is still valid
                ok = True
                if 2 * y in sums:
                    ok = False
                else:
                    for s in selected:
                        if y + s in sums:
                            ok = False
                            break
                if ok:
                    new_candidates.append(y)
            candidates = new_candidates
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size <= 16:
                break

    return best_set
