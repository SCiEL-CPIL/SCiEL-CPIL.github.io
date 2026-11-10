import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for the 60s limit
    sys.setrecursionlimit(10000)

    def get_sums(seq):
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        return sums

    # 1. Baseline Greedy
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_seq = values
    best_len = len(best_seq)
    best_sums = sums
    best_suffix = []
    for i in range(len(best_seq)):
        s = 0
        for j in range(len(best_seq)-1, i-1, -1):
            s += best_seq[j]
            if j == i:
                best_suffix.append(s)

    def check_conflict(x, current_sums, suffix_sums):
        """
        Checks if adding x to a sequence with given suffix sums causes any sum conflict.
        Returns list of new sums if valid, else None.
        """
        new_sums = [x]
        if x in current_sums:
            return None
        s = x
        for prev_s in suffix_sums:
            s += prev_s
            if s in current_sums:
                return None
            new_sums.append(s)
        return new_sums

    def dfs_inplace(current_seq, current_sums, suffix_sums, depth):
        nonlocal best_seq, best_len
        
        if time.time() > time_limit:
            return
        
        if depth > best_len:
            best_seq = current_seq[:]
            best_len = depth
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Pruning: Can we possibly beat best_len?
        # Max possible length is len(current_seq) + (256 - last_val)
        if depth + (256 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 257):
            if time.time() > time_limit:
                return
            
            new_sums = check_conflict(x, current_sums, suffix_sums)
            if new_sums is None:
                continue
                
            # Commit
            current_seq.append(x)
            for s in new_sums:
                current_sums.add(s)
            new_suffix = [x] + [x + s for s in suffix_sums]
            
            dfs_inplace(current_seq, current_sums, new_suffix, depth + 1)
            
            # Undo
            current_seq.pop()
            for s in new_sums:
                current_sums.remove(s)

    # Run a single, well-optimized DFS from the baseline
    # This explores the space of extensions from the greedy solution
    # Note: DFS from baseline is limited by the prefix. 
    # A full search from scratch is better but slower.
    # Let's try full search from scratch with aggressive pruning.
    
    def dfs_full(current_seq, current_sums, suffix_sums, depth):
        nonlocal best_seq, best_len
        
        if time.time() > time_limit:
            return
        
        if depth > best_len:
            best_seq = current_seq[:]
            best_len = depth
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Pruning
        if depth + (256 - last_val) <= best_len:
            return
            
        # Heuristic: Try smaller numbers first to keep sums small? 
        # Or larger numbers to spread them out?
        # Greedy baseline chose small numbers.
        # Let's try standard order first.
        
        for x in range(last_val + 1, 257):
            if time.time() > time_limit:
                return
            
            new_sums = check_conflict(x, current_sums, suffix_sums)
            if new_sums is None:
                continue
                
            current_seq.append(x)
            for s in new_sums:
                current_sums.add(s)
            new_suffix = [x] + [x + s for s in suffix_sums]
            
            dfs_full(current_seq, current_sums, new_suffix, depth + 1)
            
            current_seq.pop()
            for s in new_sums:
                current_sums.remove(s)

    # Start full search
    dfs_full([], set(), [], 0)

    # If time remains, try randomized restarts or different orderings?
    # The DFS above is deterministic. 
    # If it didn't find anything better, it's because the greedy prefix is a dead end for longer sequences?
    # Actually, dfs_full starts from empty, so it explores all possibilities.
    # The pruning `depth + (256 - last_val) <= best_len` is key.
    
    # If the deterministic DFS is too slow or stuck, we might want a randomized version.
    # But given the time limit, a single deep DFS is usually better than many shallow ones.
    
    return best_seq
