import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_seq = []
    
    # 1. Greedy initialization
    def greedy_smallest(deadline):
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        return values

    def _try_append(seq, sums, x):
        if seq and x <= seq[-1]:
            return False, None
        tails = [x]
        total = x
        for a in reversed(seq):
            total += a
            tails.append(total)
        for t in tails:
            if t in sums:
                return False, None
        return True, sums | set(tails)

    init_seq = greedy_smallest(deadline)
    if init_seq:
        best_seq = init_seq
        best_len = len(init_seq)
    else:
        best_len = 0

    # 2. DFS to improve
    # We will try to extend the sequence or modify it.
    # A simple DFS that tries to append elements to the current best_seq is not enough,
    # because the greedy choice might be suboptimal.
    # We should restart DFS from scratch or from different prefixes.
    
    # Let's do a full DFS with pruning.
    # To avoid re-exploring, we can use the greedy result as a lower bound.
    
    def dfs(seq, sums, deadline):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
            
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
            
        if not seq:
            last_val = 0
        else:
            last_val = seq[-1]
            
        # Pruning: Max possible length
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # Try candidates
        # Heuristic: Try smaller values first to maximize potential length
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
            
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                seq.append(x)
                dfs(seq, new_sums, deadline)
                seq.pop()
                
                # If we found a longer sequence, we can potentially prune more?
                # No, just update best.
                
    # Run DFS
    # To make it faster, we can limit the depth or use a smarter ordering.
    # Also, we can try to start DFS from the greedy solution and try to "fix" it?
    # No, standard DFS from empty is better for finding the global optimum, but might be too slow.
    # Given the small range [1, 256], the branching factor is high but depth is limited.
    # Max depth is probably around 20-30.
    # Branching factor at start is 256.
    # This is too big. We need better pruning or heuristics.
    
    # Improved Heuristic:
    # Instead of trying all x, we can use the fact that sums must be distinct.
    # If we pick x, the new sums are x, seq[-1]+x, ...
    # We can pre-filter x values that are likely to conflict.
    
    # Let's try a different approach: 
    # Use the greedy solution as a base. 
    # Then, try to replace elements in the greedy solution with larger ones to see if we can extend?
    # Or try to insert elements?
    
    # Actually, the greedy solution is usually quite good.
    # The main improvement comes from backtracking a few steps back from the greedy choice.
    
    # Let's implement a "Limited Backtracking" from the greedy solution.
    
    def limited_backtrack(init_seq, deadline):
        nonlocal best_seq, best_len
        
        sums = set()
        n = len(init_seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += init_seq[j]
                sums.add(s)
                
        best_local = list(init_seq)
        
        def dfs_local(seq, sums, idx, deadline):
            nonlocal best_local, best_len
            
            if time.time() > deadline:
                return
                
            if len(seq) > best_len:
                best_seq = list(seq)
                best_len = len(seq)
                
            if idx == len(seq):
                # Try to extend
                last_val = seq[-1] if seq else 0
                for x in range(last_val + 1, 257):
                    if time.time() > deadline:
                        return
                    ok, new_sums = _try_append(seq, sums, x)
                    if ok:
                        seq.append(x)
                        dfs_local(seq, new_sums, len(seq)-1, deadline)
                        seq.pop()
                return
                
            # Backtrack: try to replace seq[idx] with a different value
            orig_val = seq[idx]
            lb = seq[idx-1] + 1 if idx > 0 else 1
            ub = seq[idx+1] - 1 if idx < len(seq)-1 else 256
            
            # Try values in range [lb, ub]
            # To be efficient, try values close to orig_val first?
            # Or try all?
            # Let's try a window around orig_val.
            candidates = list(range(lb, ub + 1))
            # Sort by distance to orig_val?
            # candidates.sort(key=lambda v: abs(v - orig_val))
            
            for x in candidates:
                if x == orig_val:
                    continue
                if time.time() > deadline:
                    return
                    
                # Check if x is valid at position idx
                # This requires checking the whole sequence, which is expensive.
                # We can optimize by only checking sums affected by the change.
                # But for simplicity, let's just reconstruct the sums for the prefix and check.
                
                # Actually, _try_append is for appending. We need a general validity check.
                # Let's write a helper to check if a sequence is valid.
                
                trial = list(seq)
                trial[idx] = x
                
                # Check strictly increasing
                if any(trial[i+1] <= trial[i] for i in range(len(trial)-1)):
                    continue
                
                # Check distinct sums
                valid = True
                sums_trial = set()
                for i in range(len(trial)):
                    s = 0
                    for j in range(i, len(trial)):
                        s += trial[j]
                        if s in sums_trial:
                            valid = False
                            break
                        sums_trial.add(s)
                    if not valid:
                        break
                
                if valid:
                    seq[idx] = x
                    dfs_local(seq, sums_trial, idx + 1, deadline)
                    seq[idx] = orig_val
                    
        # Start from the end of the greedy sequence, backtracking a few steps
        # Try to modify the last few elements to allow a longer extension.
        
        # Let's try backtracking from the end, up to 5 steps.
        for start_idx in range(len(init_seq)-1, max(-1, len(init_seq)-6), -1):
            if time.time() > deadline:
                break
            dfs_local(list(init_seq), sums, start_idx, deadline)
            
        return best_local

    if time.time() < deadline - 5:
        limited_backtrack(init_seq, deadline)
        
    return best_seq
