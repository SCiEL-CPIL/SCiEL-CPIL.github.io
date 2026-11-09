import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Greedy construction maximizing "blocking" effect
    S = []
    U = set()  # Set of all pairwise sums a+b with a<=b in S
    candidates = set(range(1, N + 1))
    
    while candidates:
        if time.time() > deadline:
            break
            
        # 1. Identify valid candidates
        valid = []
        for x in candidates:
            # Check if x can be added to S
            # New sums: {x+s for s in S} U {2x}
            # These must be distinct and disjoint from U
            new_sums = {x + s for s in S}
            new_sums.add(2 * x)
            
            # Check distinctness within new_sums (always true if x not in S and S is Sidon? 
            # x+s1 = x+s2 => s1=s2. 2x = x+s => x=s (false). So distinctness holds.)
            
            # Check intersection with U
            if not new_sums.intersection(U):
                valid.append((x, new_sums))
        
        if not valid:
            break
            
        # 2. Calculate kill count for each valid candidate
        # To optimize, we precompute "base sums" for all candidates in valid
        # base_sums[y] = {y+s for s in S} U {2y}
        # Note: y is valid, so base_sums[y] is disjoint from U.
        
        # We want to find x in valid that kills the most y in valid\{x}
        # y is killed by x if:
        # (base_sums[y] U {y+x}) intersects (U U new_sums_x)
        # Since base_sums[y] disjoint from U, this is:
        # (base_sums[y] U {y+x}) intersects new_sums_x
        # OR (y+x) in U (which is false if y valid? No, y+x might be in U)
        # Wait, y valid means {y+s for s in S} U {2y} disjoint from U.
        # y+x is not in {y+s for s in S} unless x in S (false).
        # So y+x might be in U.
        
        # Let's refine the kill condition:
        # y is invalid w.r.t S U {x} if:
        # 1. y+x in U
        # 2. y+x in new_sums_x
        # 3. 2y in new_sums_x
        # 4. (y+s) in new_sums_x for some s in S
        
        # We precompute base_sums for all y in valid to reuse in intersection checks
        base_sums_dict = {}
        for y, _ in valid:
            bs = {y + s for s in S}
            bs.add(2 * y)
            base_sums_dict[y] = bs
            
        max_kills = -1
        best_x = None
        best_new_sums = None
        
        # Sort valid by index for determinism in ties? Or random?
        # Let's use random tie-breaking to explore different paths?
        # No, deterministic is safer for reproducibility, but "max kill" is strong.
        
        for x, new_sums_x in valid:
            kills = 0
            for y, _ in valid:
                if y == x:
                    continue
                # Check if y is killed by x
                killed = False
                
                # Condition 1: y+x in U
                if (y + x) in U:
                    killed = True
                else:
                    # Condition 2,3,4: (base_sums[y] U {y+x}) intersects new_sums_x
                    # base_sums[y] is precomputed
                    # Check intersection
                    if base_sums_dict[y].intersection(new_sums_x):
                        killed = True
                    elif (y + x) in new_sums_x:
                        killed = True
                        
                if killed:
                    kills += 1
                    
            if kills > max_kills:
                max_kills = kills
                best_x = x
                best_new_sums = new_sums_x
                # Early break if kills is maximal possible?
                # Max possible kills is len(valid) - 1
                if kills == len(valid) - 1:
                    break
                    
        if best_x is None:
            # Fallback to first valid
            best_x, best_new_sums = valid[0]
            
        # Add best_x to S
        S.append(best_x)
        U.update(best_new_sums)
        candidates.remove(best_x)
        
    return S

# EVOLVE-BLOCK-END
