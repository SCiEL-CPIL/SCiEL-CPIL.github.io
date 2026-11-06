import sys

def construct():
    # Step 1: Greedy baseline to get a good lower bound
    values_greedy = []
    sums_greedy = set()
    for x in range(1, 257):
        new_sums = []
        current_sum = 0
        for a in reversed([x] + values_greedy):
            current_sum += a
            new_sums.append(current_sum)
        
        if not sums_greedy.intersection(new_sums):
            values_greedy.append(x)
            sums_greedy.update(new_sums)
    
    best_len = len(values_greedy)
    best_seq = values_greedy[:]
    
    # Step 2: Backtracking CSP Solver
    # State:
    # - seq: list of chosen integers
    # - prefix_sums: list of cumulative sums p_0=0, p_1, ..., p_k
    # - diffs: set of all differences p_j - p_i for i < j
    
    # Initialize with empty sequence
    # We use in-place modification for efficiency
    
    seq = []
    prefix_sums = [0]
    diffs = set()
    
    # Optimization: Pre-calculate or just use set operations
    # We need to check if new_prefix creates any diff in diffs
    
    def dfs():
        nonlocal best_len, best_seq
        
        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_seq = seq[:]
            
        if current_len == 256:
            return
            
        last_val = seq[-1] if seq else 0
        last_prefix = prefix_sums[-1]
        
        # Pruning: Max possible additional elements
        # Since values must be strictly increasing integers in [1, 256]
        # The smallest possible next value is last_val + 1
        # The largest possible next value is 256
        # So max additional elements is 256 - last_val
        if current_len + (256 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 257):
            new_prefix = last_prefix + x
            
            # Check for collisions
            # New differences are new_prefix - p_i for all i in 0..k
            # We must ensure none of these are in 'diffs'
            collision = False
            for p in prefix_sums:
                d = new_prefix - p
                if d in diffs:
                    collision = True
                    break
            
            if not collision:
                # Add x
                seq.append(x)
                prefix_sums.append(new_prefix)
                
                # Add new diffs to set
                new_diffs = []
                for p in prefix_sums[:-1]: # exclude new_prefix itself? 
                    # diff is new_prefix - p
                    # We already checked these are not in diffs.
                    # We need to add them to diffs for future steps.
                    new_diffs.append(new_prefix - p)
                
                diffs.update(new_diffs)
                
                # Recurse
                dfs()
                
                # Backtrack
                seq.pop()
                prefix_sums.pop()
                for d in new_diffs:
                    diffs.discard(d)
                    
    dfs()
    
    return best_seq

