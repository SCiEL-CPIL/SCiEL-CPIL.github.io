import time

def construct():
    start_time = time.time()
    # 90s limit, use 85s to be safe
    deadline = start_time + 85.0
    
    best_seq = []
    
    # State
    seq = []
    sums = set()
    suffix_sums = [] # List of sums of consecutive blocks ending at the last element
    
    def dfs(last_val):
        nonlocal best_seq, deadline, seq, sums, suffix_sums
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        # Pruning: Even if we add the smallest possible values (last+1, last+2, ...),
        # can we beat the best?
        # Max number of additional values is 256 - last_val
        # This is a very loose bound, but cheap.
        if len(seq) + (256 - last_val) <= len(best_seq):
            return

        # Try candidates
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Compute new suffix sums
            # New suffix sums are x, x + old_suffix_1, x + old_suffix_2, ...
            new_suffix_sums = [x]
            # Optimization: Check conflict while building
            conflict = False
            if x in sums:
                conflict = True
            else:
                for s in suffix_sums:
                    ns = x + s
                    new_suffix_sums.append(ns)
                    if ns in sums:
                        conflict = True
                        break
            
            if conflict:
                continue
                
            # Valid
            seq.append(x)
            sums.update(new_suffix_sums)
            old_suffix_sums = suffix_sums
            suffix_sums = new_suffix_sums
            
            dfs(x)
            
            # Backtrack
            seq.pop()
            sums.difference_update(new_suffix_sums)
            suffix_sums = old_suffix_sums

    dfs(0)
    return best_seq
