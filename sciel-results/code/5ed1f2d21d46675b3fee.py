def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(2, 41))
    best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Try to find 18
    # Allocate 80% of time to 18
    time_for_18 = 68.0
    
    for r in range(10000):
        if time.time() > start_time + time_for_18:
            break
            
        S = []
        P = {1}
        valid = True
        
        for step in range(18):
            valid_cands = []
            for x in candidates:
                if x in S:
                    continue
                # Check collision
                # new_prods = {p * x for p in P}
                # if new_prods & P:
                #    collision
                # Optimization: if x is in P, then 1*x=x is in P. Collision.
                # So x not in P is a necessary condition.
                if x in P:
                    continue
                    
                # Generate new products
                # This is the expensive part
                new_prods = set()
                for p in P:
                    new_prods.add(p * x)
                    
                if new_prods & P:
                    continue
                    
                valid_cands.append(x)
                
            if not valid_cands:
                valid = False
                break
                
            x = random.choice(valid_cands)
            S.append(x)
            
            # Update P
            # P = P | new_prods
            # But we discarded new_prods. We can recompute or store.
            # To save time, we can just update P in the next iteration's candidate check?
            # No, we need P for the next step.
            # Let's recompute P update efficiently.
            # Actually, we already computed new_prods for the chosen x?
            # No, we computed it for ALL candidates.
            # We should only compute for the chosen one?
            # But we need to find valid candidates first.
            
            # Better: 
            # 1. Find all valid candidates.
            # 2. Pick one.
            # 3. Update P using the precomputed new_prods for that x?
            # We didn't store them.
            
            # Alternative:
            # Just update P by iterating P again.
            # P_new = P | {p*x for p in P}
            # This is O(2^k).
            
            P.update([p * x for p in P])
            
        if valid:
            return S
            
    # If 18 not found, try 17 with remaining time
    time_for_17 = deadline - time.time()
    
    for r in range(10000):
        if time.time() > deadline:
            break
            
        S = []
        P = {1}
        valid = True
        
        for step in range(17):
            valid_cands = []
            for x in candidates:
                if x in S:
                    continue
                if x in P:
                    continue
                new_prods = set()
                for p in P:
                    new_prods.add(p * x)
                if new_prods & P:
                    continue
                valid_cands.append(x)
                
            if not valid_cands:
                valid = False
                break
                
            x = random.choice(valid_cands)
            S.append(x)
            P.update([p * x for p in P])
            
        if valid and len(S) > len(best_set):
            best_set = S
            
    return best_set
