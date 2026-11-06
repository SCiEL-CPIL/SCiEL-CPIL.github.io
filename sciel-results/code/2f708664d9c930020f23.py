import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_seq = []
    best_len = 0

    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()

    # Helper to check if a sequence is valid (all consecutive block sums distinct)
    def is_valid(seq):
        if not seq:
            return True
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Greedy Initialization
    def greedy_init():
        used = 0
        suffix_mask = 0
        seq = []
        for x in range(1, 257):
            if time.time() > deadline:
                break
            N = (1 << x) | (suffix_mask << x)
            if (N & used) == 0:
                used |= N
                suffix_mask = N
                seq.append(x)
        return seq

    # Initial greedy
    init_seq = greedy_init()
    update_best(init_seq)

    # DFS from scratch
    def dfs(last_val, used, suffix_mask, seq):
        nonlocal best_seq, best_len
        
        current_len = len(seq)
        
        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()
        
        if time.time() > deadline:
            return
        
        if last_val >= 256:
            return
        
        # Upper bound
        if current_len + (256 - last_val) <= best_len:
            return
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            N = (1 << x) | (suffix_mask << x)
            if (N & used) == 0:
                new_used = used | N
                new_suffix_mask = N
                seq.append(x)
                dfs(x, new_used, new_suffix_mask, seq)
                seq.pop()

    dfs(0, 0, 0, [])

    # DFS Extend from best sequence found so far
    if best_len > 0:
        last_val = best_seq[-1]
        used = 0
        suffix_mask = 0
        for x in best_seq:
            N = (1 << x) | (suffix_mask << x)
            used |= N
            suffix_mask = N
        
        def dfs_extend(last_val, used, suffix_mask, seq):
            nonlocal best_seq, best_len
            
            current_len = len(seq)
            
            if current_len > best_len:
                best_len = current_len
                best_seq = seq.copy()
            
            if time.time() > deadline:
                return
            
            if last_val >= 256:
                return
            
            if current_len + (256 - last_val) <= best_len:
                return
            
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                
                N = (1 << x) | (suffix_mask << x)
                if (N & used) == 0:
                    new_used = used | N
                    new_suffix_mask = N
                    seq.append(x)
                    dfs_extend(x, new_used, new_suffix_mask, seq)
                    seq.pop()

        dfs_extend(last_val, used, suffix_mask, best_seq.copy())

    # Local Search on best result
    if time.time() < deadline:
        def local_search(seq):
            nonlocal best_seq, best_len
            improved = True
            while improved and time.time() < deadline:
                improved = False
                n = len(seq)
                for pos in range(n + 1):
                    if time.time() > deadline:
                        return seq
                    for x in range(1, 257):
                        if x in seq:
                            continue
                        if pos > 0 and x <= seq[pos - 1]:
                            continue
                        if pos < n and x >= seq[pos]:
                            continue
                        
                        new_seq = seq[:pos] + [x] + seq[pos:]
                        if is_valid(new_seq):
                            seq = new_seq
                            improved = True
                            if len(seq) > best_len:
                                best_len = len(seq)
                                best_seq = seq.copy()
                            break
            return seq

        local_search(best_seq.copy())

    return best_seq
