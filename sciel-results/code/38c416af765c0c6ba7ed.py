def construct():
    best_seq = []
    best_len = 0
    
    # candidates are 2 to 128
    candidates = list(range(2, 129))
    
    def dfs(idx, seq, prods):
        nonlocal best_seq, best_len
        
        # Pruning: even if we take all remaining, can we beat best?
        if len(seq) + (len(candidates) - idx) <= best_len:
            return
            
        if idx == len(candidates):
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            return
            
        x = candidates[idx]
        
        # Option 1: Try to include x
        # Calculate new products
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(seq):
            prod *= a
            new_prods.append(prod)
            
        # Check conflict
        conflict = False
        for p in new_prods:
            if p in prods:
                conflict = True
                break
                
        if not conflict:
            # Include x
            new_prods_set = set(prods)
            new_prods_set.update(new_prods)
            seq.append(x)
            dfs(idx + 1, seq, new_prods_set)
            seq.pop()
            
        # Option 2: Skip x
        dfs(idx + 1, seq, prods)

    dfs(0, [], set())
    return best_seq
