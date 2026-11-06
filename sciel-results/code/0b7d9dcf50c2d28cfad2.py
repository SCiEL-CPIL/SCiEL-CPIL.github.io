import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    best_seq = []
    best_len = 0
    
    # Greedy initialization - try both ascending and descending, keep the longer
    def greedy_asc():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals
    
    def greedy_desc():
        vals = []
        sums_set = set()
        for x in range(256, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        # Return in increasing order
        vals.sort()
        return vals
    
    seq_asc = greedy_asc()
    seq_desc = greedy_desc()
    
    if len(seq_desc) > len(seq_asc):
        initial_seq = seq_desc
    else:
        initial_seq = seq_asc
    
    if initial_seq:
        best_seq = initial_seq.copy()
        best_len = len(initial_seq)
    
    # Compute suffix sums for a sequence
    def compute_suffix_sums(seq):
        n = len(seq)
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + seq[i]
        return suffix
    
    # Compute sums mask from a sequence
    def calc_sums_mask(seq):
        mask = 0
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                mask |= (1 << s)
        return mask
    
    # DFS with suffix sums maintained - try LARGER candidates first
    def dfs(seq, sums_mask, last_val, suffix_sums):
        nonlocal best_seq, best_len, start_time, time_limit
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
            
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Pruning: if even using all remaining values can't beat best, prune
        remaining = 256 - last_val
        if current_len + remaining <= best_len:
            return
        
        # Try to extend: candidates from last_val + 1 to 256
        # Try LARGER candidates first (they push sums higher, reducing collisions)
        for x in range(256, last_val, -1):
            # Check time periodically
            if (256 - x) % 16 == 0 and time.time() - start_time > time_limit:
                return
            
            # Compute new sums mask efficiently
            new_sums_mask = 0
            total = x
            new_sums_mask |= (1 << total)
            for i in range(len(seq) - 1, -1, -1):
                total += seq[i]
                new_sums_mask |= (1 << total)
            
            if sums_mask & new_sums_mask:
                continue
            
            # Build new suffix sums
            new_suffix = [0] * (len(seq) + 2)
            new_suffix[len(seq) + 1] = 0
            new_suffix[len(seq)] = x
            for i in range(len(seq) - 1, -1, -1):
                new_suffix[i] = suffix_sums[i] + x
            
            dfs(seq + [x], sums_mask | new_sums_mask, x, new_suffix)
            
            # Early termination if time is up
            if time.time() - start_time > time_limit:
                return
    
    # Start DFS from empty
    dfs([], 0, 0, [0])
    
    return best_seq
