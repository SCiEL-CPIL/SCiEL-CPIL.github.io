import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_seq = []
    best_len = 0

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

    # Helper to compute all consecutive block sums for a sequence
    def all_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    # Helper to update best
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()

    # 1. Greedy Initialization
    used = 0
    suffix_mask = 0
    greedy_seq = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if (used >> x) & 1:
            continue
        if (used >> x) & suffix_mask:
            continue
        
        greedy_seq.append(x)
        used |= (1 << x) | (suffix_mask << x)
        suffix_mask = (1 << x) | (suffix_mask << x)

    update_best(greedy_seq)

    # 2. Local Search Insertion: try to insert elements into the best sequence
    def local_search_insertion(seq):
        nonlocal best_seq, best_len
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(seq)
            # Try inserting each candidate at each position
            for pos in range(n + 1):
                if time.time() > deadline:
                    return
                for x in range(1, 257):
                    if x in seq:
                        continue
                    # Check if x is strictly between neighbors
                    if pos > 0 and x <= seq[pos - 1]:
                        continue
                    if pos < n and x >= seq[pos]:
                        continue
                    
                    # Try inserting x at position pos
                    new_seq = seq[:pos] + [x] + seq[pos:]
                    if is_valid(new_seq):
                        seq = new_seq
                        improved = True
                        if len(seq) > best_len:
                            best_len = len(seq)
                            best_seq = seq.copy()
                        break  # Move to next position
        return seq

    # Apply local search to greedy result
    if time.time() < deadline:
        ls_seq = local_search_insertion(greedy_seq.copy())
        update_best(ls_seq)

    # 3. DFS with Backtracking
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
        
        if current_len + (256 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            if (used >> x) & 1:
                continue
            if (used >> x) & suffix_mask:
                continue
            
            seq.append(x)
            dfs(x, used | (1 << x) | (suffix_mask << x), 
                (1 << x) | (suffix_mask << x), seq)
            seq.pop()

    # Run DFS
    dfs(0, 0, 0, [])

    # 4. Final local search on best result found by DFS
    if time.time() < deadline:
        final_ls = local_search_insertion(best_seq.copy())
        update_best(final_ls)

    return best_seq
