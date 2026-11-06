def construct():
    from math import lcm
    
    n = 512
    # Precompute LCMs? No, just calculate on fly, it's fast enough for 512^2
    # But we will do many checks. Let's precompute a compatibility matrix or just use lcm.
    # 512*512 = 262k pairs. Precomputing is fast.
    
    # Actually, we can just use math.lcm.
    
    def is_compatible(x, y):
        return lcm(x, y) <= n

    # 1. Initial Greedy
    selected = [1]
    for x in range(2, n + 1):
        compatible = True
        for y in selected:
            if lcm(x, y) > n:
                compatible = False
                break
        if compatible:
            selected.append(x)
    
    S = set(selected)
    
    # 2. Local Search / Iterative Improvement
    # We will try to add all possible elements, then if stuck, remove one and try again.
    
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Add all compatible elements
        candidates = [x for x in range(1, n + 1) if x not in S]
        addable = []
        for x in candidates:
            ok = True
            for y in S:
                if lcm(x, y) > n:
                    ok = False
                    break
            if ok:
                addable.append(x)
        
        if addable:
            S.update(addable)
            improved = True
            # Loop again to see if adding these opens up more? 
            # Actually, adding elements only *increases* constraints, so it won't open up new ones 
            # that were previously blocked by other elements. 
            # Wait, if x was blocked by y, and y is still in S, x is still blocked.
            # So adding elements never makes a previously blocked element unblocked.
            # So we only need to do this phase once per "removal" cycle?
            # Yes.
            continue
        
        # Phase 2: If no addable elements, try to remove one element to allow more additions
        best_gain = 0
        best_remover = None
        best_new_adds = None
        
        # To speed up, we can sample or just check all. |S| is likely < 100.
        for y in list(S):
            S_temp = S - {y}
            # Find how many candidates become addable
            # Note: Some candidates were already addable? No, we established none were.
            # So any addable candidate in S_temp is a NEW addition.
            new_adds = []
            for x in candidates: # candidates is the set of elements NOT in S (original)
                # Wait, candidates list was built based on S.
                # If we remove y, the set of non-S elements changes? No, S_temp is smaller, 
                # so the set of elements NOT in S_temp is larger (includes y).
                # But we don't want to re-add y immediately unless it helps? 
                # Actually, if we remove y, we can potentially add y back later if it doesn't block others?
                # No, if y was removed because it blocked others, re-adding it might block them again.
                # We should look for x in [1..n] \ S_temp such that x != y (usually) 
                # and compatible with S_temp.
                
                # Optimization: Only check x that were previously blocked by y?
                # Simple approach: Check all x in [1..n] not in S_temp.
                ok = True
                for z in S_temp:
                    if lcm(x, z) > n:
                        ok = False
                        break
                if ok:
                    new_adds.append(x)
            
            gain = len(new_adds) - 1 # Removed 1, added len(new_adds)
            if gain > best_gain:
                best_gain = gain
                best_remover = y
                best_new_adds = new_adds
        
        if best_gain > 0:
            S.remove(best_remover)
            S.update(best_new_adds)
            improved = True
            # Note: We should re-evaluate "candidates" list in next iteration?
            # The `candidates` variable in the loop scope is stale.
            # But we `continue` to the top of the while loop, which rebuilds `candidates`.
            # Wait, the `candidates` list is built at the start of the while loop iteration.
            # So yes, it will be refreshed.
            continue
        else:
            break
            
    return list(S)
