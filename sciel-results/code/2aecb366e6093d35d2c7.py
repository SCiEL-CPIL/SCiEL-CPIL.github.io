import time

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Greedy initialization to get a good lower bound
    def greedy():
        values = []
        sums = set()
        for x in range(1, N + 1):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    init_seq, init_sums = greedy()
    best_seq = init_seq
    best_len = len(init_seq)
    
    # Calculate suffix sums for init_seq to be consistent with DFS state
    # But we start DFS from scratch to explore other branches.
    # We just need the initial best for pruning.
    
    def dfs(start_val, seq, suffix_sums, sums_set):
        nonlocal best_seq, best_len
        
        # Pruning: if we can't beat the best length, return
        if len(seq) + (N - start_val + 1) <= best_len:
            return
        
        # Time check
        if time.time() > deadline:
            return
        
        # Iterate candidates
        # We can try to order candidates. 
        # Sometimes trying larger numbers first helps find long sequences faster?
        # Or smaller? 
        # Let's stick to increasing order for simplicity, but we can reverse if needed.
        # Actually, if we find a long sequence early, we prune more.
        
        for x in range(start_val, N + 1):
            # Calculate new suffix sums
            # suffix_sums contains sums of suffixes of seq: [s1, s2, ..., sk]
            # s1 is sum of last 1, sk is sum of all.
            # New suffixes: [x, x+s1, x+s2, ..., x+sk]
            
            # Optimization: Check collisions before creating new list/set
            # But we need the new sums for the set anyway if valid.
            
            new_suffix_sums = [x]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            valid = True
            for s in new_suffix_sums:
                if s in sums_set:
                    valid = False
                    break
            
            if valid:
                new_sums_set = sums_set.union(new_suffix_sums)
                new_seq = seq + [x]
                
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq[:]
                
                # Recurse
                dfs(x + 1, new_seq, new_suffix_sums, new_sums_set)

    dfs(1, [], [], set())
    
    return best_seq
