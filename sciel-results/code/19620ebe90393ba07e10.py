import time

def construct():
    start_time = time.time()
    time_limit = 58.0
    
    best_len = 0
    best_seq = []
    
    def dfs(prefix_sums, forbidden_sums, last_val, current_seq):
        nonlocal best_len, best_seq
        nonlocal start_time, time_limit
        
        if time.time() > time_limit:
            return
        
        # Pruning
        if len(current_seq) + (256 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 257):
            if time.time() > time_limit:
                return
            
            new_prefix = prefix_sums[-1] + x
            valid = True
            new_sums = []
            for p in prefix_sums:
                s = new_prefix - p
                if s in forbidden_sums:
                    valid = False
                    break
                new_sums.append(s)
                
            if valid:
                # Commit
                prefix_sums.append(new_prefix)
                for s in new_sums:
                    forbidden_sums.add(s)
                current_seq.append(x)
                
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq.copy()
                    
                dfs(prefix_sums, forbidden_sums, x, current_seq)
                
                # Undo
                current_seq.pop()
                for s in new_sums:
                    forbidden_sums.remove(s)
                prefix_sums.pop()
                
                if best_len == 256:
                    return

    prefix_sums = [0]
    forbidden_sums = set()
    last_val = 0
    current_seq = []
    
    dfs(prefix_sums, forbidden_sums, last_val, current_seq)
    
    return best_seq
