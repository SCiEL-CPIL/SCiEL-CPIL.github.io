import time

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin before 60s limit
    
    best_seq = []
    
    # Initial greedy pass to get a decent starting point? 
    # Or just start DFS from empty?
    # Starting from empty with bitsets might be fast enough for small N.
    # But to be safe, let's do a quick greedy to initialize best_seq.
    
    def greedy_init():
        vals = []
        sums = set()
        for x in range(1, 257):
            # Check if x can be appended
            # New sums: x, x+last, x+last+prev...
            new_sums = [x]
            total = x
            for a in reversed(vals):
                total += a
                new_sums.append(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        return vals

    initial_seq = greedy_init()
    if initial_seq:
        best_seq = initial_seq.copy()
        
    # Calculate sums mask for initial sequence
    def calc_mask(seq):
        mask = 0
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                mask |= (1 << s)
        return mask

    # If we start DFS from greedy result, we need to ensure we can extend it.
    # However, the greedy result might be a dead end for further extension if we just append.
    # But we can try to extend it.
    # Better: Start DFS from scratch, but use the greedy length as a lower bound for pruning?
    # No, DFS from scratch explores all possibilities.
    # Given the constraint [1, 256] and distinct block sums, the max length is likely around 20-30.
    # The search space is manageable with bitsets.
    
    # Let's restart DFS from empty to find the global optimum, 
    # but use the time limit.
    
    def dfs(seq, sums_mask, last_val):
        nonlocal best_seq
        nonlocal start_time
        nonlocal time_limit
        
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Try to extend
        # Candidates: last_val + 1 to 256
        # Optimization: If we are already longer than best, we don't need to prune by length, 
        # but we should still prune by collisions.
        
        # To speed up, we can try smaller candidates first? 
        # Or larger? 
        # Smaller candidates keep the sum lower, potentially allowing more elements.
        # Let's try from last_val + 1 upwards.
        
        for x in range(last_val + 1, 257):
            # Calculate new sums mask
            # New sums are x, x+seq[-1], x+seq[-1]+seq[-2], ...
            # Let suffix_sums be the list of suffix sums of seq.
            # We can compute them on the fly or maintain them.
            # Maintaining them is O(1) per step if we pass them.
            # Let's pass suffix_sums.
            
            # But in this simple dfs, we just compute.
            new_sums_mask = 1 << x
            total = x
            for a in reversed(seq):
                total += a
                new_sums_mask |= (1 << total)
                
            if sums_mask & new_sums_mask:
                continue
                
            dfs(seq + [x], sums_mask | new_sums_mask, x)
            
            # Pruning: If we found a sequence longer than best, 
            # and we are trying smaller x, maybe larger x won't help? 
            # Not necessarily.
            
            # Early termination if time is up
            if time.time() - start_time > time_limit:
                return

    # Start DFS
    # To make it faster, we can use the greedy result as a guide?
    # No, pure DFS with bitsets is strong.
    # However, the branching factor is up to 256.
    # Depth is ~20.
    # 256^20 is huge. Pruning is essential.
    # The bitset check is O(1) effectively (big int op).
    
    # Let's add a simple heuristic: 
    # If the current length + remaining possible values < len(best_seq), prune.
    # Remaining possible values is at most 256 - last_val.
    # This is a weak bound.
    
    # A stronger bound: 
    # The minimum possible sum of the next k elements is (last+1) + ... 
    # This doesn't help with distinct sums directly.
    
    # Let's just run the DFS. The bitset pruning is very effective because 
    # collisions happen frequently.
    
    dfs([], 0, 0)
    
    return best_seq
