def construct():
    # Initial Greedy
    S = []
    S_3_sums = set()
    rejected = []
    
    # Order: Descending
    for x in range(192, 0, -1):
        valid = True
        
        # Check 1: x divides existing sum
        for s in S_3_sums:
            if s % x == 0:
                valid = False
                break
        
        if not valid:
            rejected.append(x)
            continue
            
        # Check 2: existing a divides new sum with x
        for a in S:
            found_conflict = False
            for i in range(len(S)):
                u = S[i]
                if u == a: continue
                for j in range(i + 1, len(S)):
                    v = S[j]
                    if v == a: continue
                    if (x + u + v) % a == 0:
                        found_conflict = True
                        break
                if found_conflict: break
            if found_conflict:
                valid = False
                break
        
        if valid:
            S.append(x)
            old_S = S[:-1]
            for i in range(len(old_S)):
                u = old_S[i]
                for j in range(i + 1, len(old_S)):
                    v = old_S[j]
                    S_3_sums.add(x + u + v)
        else:
            rejected.append(x)
            
    # Local Search
    # We will try to improve S by swapping elements from S with elements from rejected
    # To keep it simple and fast, we do a few passes
    
    current_best = len(S)
    
    # Helper to check validity of a set
    def is_valid_set(set_S):
        if len(set_S) < 4:
            return True
        sums = set()
        for i in range(len(set_S)):
            for j in range(i+1, len(set_S)):
                for k in range(j+1, len(set_S)):
                    sums.add(set_S[i] + set_S[j] + set_S[k])
        
        for x in set_S:
            for s in sums:
                # s is sum of 3 distinct others?
                # The set `sums` contains sums of ALL triples in set_S.
                # If x is part of a triple, its sum is in `sums`.
                # But we need sums of 3 DISTINCT OTHERS.
                # So we need to ensure the triple does not contain x.
                # This simple check `s % x == 0` might be true for a sum containing x.
                # E.g., x=1, sum=1+2+3=6. 6%1==0. But 1 is in the sum.
                # So we need to be careful.
                
                pass # This helper is too slow/complex for inline. 
                # Better to use the incremental check from the greedy.
        
        return True # Placeholder
    
    # Instead of a generic is_valid, we will use a targeted swap check.
    # If we swap x_out with x_in, we only need to check conflicts involving x_in 
    # and the changed sums.
    
    # Given time constraints, I will implement a simpler local search:
    # Try to add rejected elements. If conflict, try removing a conflicting element.
    
    # Re-implementing the local search loop
    # To avoid code duplication, I'll write a function to check if x can be added to S
    # and return the conflicting elements if it can't.
    
    def check_add(x, S_list, S_sums):
        # Returns (is_valid, conflicting_elements)
        # conflicting_elements is a list of elements in S_list that prevent x from being added
        
        # 1. x divides existing sum
        for s in S_sums:
            if s % x == 0:
                # We need to identify which triple formed s.
                # We didn't store the triples.
                # This makes it hard to pick which element to remove.
                # Heuristic: Remove a random element from S?
                # Or, just try to remove each element in S one by one and see if it works?
                # That's O(|S|) checks.
                return False, [] # Simplified: just say invalid
        
        # 2. Existing a divides new sum
        for a in S_list:
            for i in range(len(S_list)):
                u = S_list[i]
                if u == a: continue
                for j in range(i + 1, len(S_list)):
                    v = S_list[j]
                    if v == a: continue
                    if (x + u + v) % a == 0:
                        return False, [a, u, v]
                        
        return True, []
        
    # This local search is getting complex. 
    # Given the 60s limit and Python speed, a simple improvement might be:
    # Run the greedy multiple times with different orderings and keep the best.
    
    best_S = S
    best_len = len(S)
    
    import random
    random.seed(42)
    
    for _ in range(20):
        S_temp = []
        S_3_sums_temp = set()
        order = list(range(192, 0, -1))
        random.shuffle(order)
        
        for x in order:
            valid = True
            for s in S_3_sums_temp:
                if s % x == 0:
                    valid = False
                    break
            if not valid:
                continue
            for a in S_temp:
                found = False
                for i in range(len(S_temp)):
                    u = S_temp[i]
                    if u == a: continue
                    for j in range(i + 1, len(S_temp)):
                        v = S_temp[j]
                        if v == a: continue
                        if (x + u + v) % a == 0:
                            found = True
                            break
                    if found: break
                if found:
                    valid = False
                    break
            if valid:
                S_temp.append(x)
                old = S_temp[:-1]
                for i in range(len(old)):
                    u = old[i]
                    for j in range(i + 1, len(old)):
                        v = old[j]
                        S_3_sums_temp.add(x + u + v)
                        
        if len(S_temp) > best_len:
            best_len = len(S_temp)
            best_S = S_temp
            
    return best_S
