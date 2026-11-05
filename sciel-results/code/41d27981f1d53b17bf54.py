import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Initial set: upper tail
    # 3a > 2n => a > 170.66 => a >= 171
    S = set(range(171, n + 1))
    best_S = set(S)
    best_size = len(S)
    
    # Precompute conflicts for speed? 
    # For a given a, b, c, check a | (b+c)
    
    def is_valid(S):
        # Check all triples a < b <= c in S
        # This is O(|S|^3) which is ~ 86^3 = 635k ops. Fast enough.
        # But we do this in local search, so need to be careful.
        # We can check incrementally.
        pass

    def check_addition(S, x):
        # Check if adding x to S makes it invalid
        # Only need to check triples involving x
        # x can be a, b, or c
        # Case 1: x is a. x < b <= c. x | (b+c)
        # Case 2: x is b. a < x <= c. a | (x+c)
        # Case 3: x is c. a < b <= x. a | (b+x)
        
        # We need to find if ANY such triple exists.
        # If yes, return list of conflicting (b,c) or (a,c) or (a,b)
        
        conflicts = []
        
        # Case 1: x is a
        # x < b <= c, b,c in S
        # x | (b+c)
        for b in S:
            if b <= x: continue
            for c in S:
                if c < b: continue
                if (b + c) % x == 0:
                    conflicts.append(('a', b, c))
                    # Early exit? No, we might want to remove minimal set.
                    # But for now, just check existence.
        
        # Case 2: x is b
        # a < x <= c, a,c in S
        # a | (x+c)
        for a in S:
            if a >= x: continue
            for c in S:
                if c < x: continue
                if (x + c) % a == 0:
                    conflicts.append(('b', a, c))
        
        # Case 3: x is c
        # a < b <= x, a,b in S
        # a | (b+x)
        for a in S:
            if a >= x: continue
            for b in S:
                if b <= a: continue
                if b > x: continue
                if (b + x) % a == 0:
                    conflicts.append(('c', a, b))
        
        return conflicts

    # Local Search
    # We try to add elements from 1 to 170
    candidates = list(range(1, 171))
    random.shuffle(candidates)
    
    # We also try to swap: remove one, add one
    # But adding without removal is best.
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in S:
            continue
            
        conflicts = check_addition(S, x)
        
        if not conflicts:
            S.add(x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # Try to resolve by removing elements
            # Identify elements involved in conflicts
            # If we remove one element, does it clear all conflicts?
            # This is a hitting set problem.
            # Simple heuristic: try removing each element in S that is in a conflict
            # and see if the set becomes valid with x added.
            
            involved = set()
            for conf in conflicts:
                involved.add(conf[1])
                involved.add(conf[2])
            
            # Try removing one element from involved
            for rem in involved:
                if time.time() > deadline:
                    break
                S_new = S - {rem}
                S_new.add(x)
                # Check if S_new is valid
                # We only need to check if the conflicts are gone and no new ones created?
                # New conflicts could be created if rem was masking something?
                # No, removing an element only removes triples.
                # So if the conflicts involving x are gone, and x didn't create conflicts with the rest, it's valid.
                # Wait, check_addition checked x vs S.
                # If we remove rem, we need to check if x vs (S - {rem}) is valid.
                # And also, did rem's removal create issues? No, removal is safe.
                
                # So just check if x is valid with S - {rem}
                # We can reuse check_addition logic but it's O(|S|^2).
                # Let's just check if any conflict remains.
                
                valid = True
                # Re-check conflicts with S_new
                # Optimization: only check triples involving x
                # Case 1: x is a
                for b in S_new:
                    if b <= x: continue
                    for c in S_new:
                        if c < b: continue
                        if (b + c) % x == 0:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    # Case 2: x is b
                    for a in S_new:
                        if a >= x: continue
                        for c in S_new:
                            if c < x: continue
                            if (x + c) % a == 0:
                                valid = False
                                break
                        if not valid: break
                if valid:
                    # Case 3: x is c
                    for a in S_new:
                        if a >= x: continue
                        for b in S_new:
                            if b <= a: continue
                            if b > x: continue
                            if (b + x) % a == 0:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    # We added 1, removed 1. Net 0.
                    # This might help future additions.
                    # Accept with some probability? Or always accept if size is same?
                    # Let's accept to explore.
                    S = S_new
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break # Try next candidate x? Or keep trying to improve this x?
            
            # If we couldn't resolve with 1 removal, try 2 removals?
            # Too slow.
            
    return list(best_S)
