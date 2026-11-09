import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin to ensure finish within 90s
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, sums, next_val):
        nonlocal best_seq, best_len
        
        # Check time
        if time.time() - start_time > time_limit:
            return
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: if even adding all remaining numbers won't beat best_len, stop
        # Max possible length from here is len(seq) + (256 - next_val + 1)
        if len(seq) + (256 - next_val + 1) <= best_len:
            return

        # Try each candidate x
        for x in range(next_val, 257):
            # Check time periodically
            if time.time() - start_time > time_limit:
                return

            # Calculate new sums
            new_sums = []
            curr_sum = 0
            for a in reversed(seq):
                curr_sum += a
                new_sums.append(curr_sum + x)
            new_sums.append(x)
            
            # Check for intersection
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
            
            if not conflict:
                # Recurse
                new_sums_set = sums.union(new_sums)
                dfs(seq + [x], new_sums_set, x + 1)
                
                # Pruning in loop: if we can't beat best_len with remaining numbers
                # If we pick x, we use 1 slot. Remaining slots available: 256 - x.
                # Total potential length: len(seq) + 1 + (256 - x)
                if len(seq) + 1 + (256 - x) <= best_len:
                    break

    dfs([], set(), 1)
    
    return best_seq
