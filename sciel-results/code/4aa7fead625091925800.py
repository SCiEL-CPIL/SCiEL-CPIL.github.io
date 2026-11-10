import time

def construct():
    """
    Returns a strictly increasing sequence of integers in [1, 256] 
    such that sums of all nonempty consecutive blocks are distinct.
    Maximizes sequence length.
    """
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, all_sums, suffix_sums):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Update best if current sequence is longer
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        last_val = seq[-1] if seq else 0
        
        # Try candidates in increasing order to keep sums small
        for x in range(last_val + 1, 257):
            # Time check
            if time.time() - start_time > time_limit:
                return
            
            # Generate new suffix sums
            # New suffixes are x and x + s for all s in suffix_sums
            new_suffixes = [x]
            for s in suffix_sums:
                new_suffixes.append(x + s)
            
            # Check for collisions with existing all_sums
            valid = True
            for s in new_suffixes:
                if s in all_sums:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Recurse with new state
            new_all_sums = all_sums.union(new_suffixes)
            new_suffix_set = set(new_suffixes)
            
            dfs(seq + [x], new_all_sums, new_suffix_set)
            
    dfs([], set(), set())
    
    return best_seq
