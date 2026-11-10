import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Get a good baseline using the greedy approach from the seed
    best_seq, best_products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(best_seq):
            prod *= a
            tails.append(prod)
        if not best_products.intersection(tails):
            best_seq.append(x)
            best_products.update(tails)
    
    best_len = len(best_seq)
    candidates = list(range(2, 129))
    n_candidates = len(candidates)
    
    # 2. DFS to try to improve
    # We use a list to store the current state to avoid copying lists excessively?
    # Given n=127, copying lists of size <=127 is cheap.
    
    def dfs(idx, seq, prods):
        nonlocal best_len, best_seq, best_products
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: Even if we take all remaining, can we beat best_len?
        if len(seq) + (n_candidates - idx) <= best_len:
            return
        
        if idx == n_candidates:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq.copy()
                best_products = prods.copy()
            return
        
        x = candidates[idx]
        
        # Option 1: Include x
        # Calculate new tails
        new_tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            new_tails.append(prod)
            
        # Check collision
        # Optimization: Check intersection
        if not prods.intersection(new_tails):
            new_prods = prods.union(new_tails)
            # Recurse
            dfs(idx + 1, seq + [x], new_prods)
            
        # Option 2: Exclude x
        # We can prune this branch if including x didn't help and we are far from best?
        # No, excluding might allow a better combination later?
        # Actually, if we exclude x, we lose the chance to use x.
        # But maybe x blocks a later number?
        # Yes, so we must explore Exclude.
        
        # Heuristic: If we just updated best_len in the Include branch, 
        # the Exclude branch is less likely to beat it unless it has much more room.
        # But we still need to check.
        
        # Check time again
        if time.time() > deadline:
            return
            
        dfs(idx + 1, seq, prods)

    dfs(0, [], set())
    
    return best_seq
