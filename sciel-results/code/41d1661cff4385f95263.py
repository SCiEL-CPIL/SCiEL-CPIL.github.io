import time
import sys

sys.setrecursionlimit(10000)

def _greedy_largest():
    """
    Greedy strategy: always pick the largest valid number.
    """
    seq = []
    sums = set()
    for x in range(256, 0, -1):
        if seq and x <= seq[-1]:
            continue
        s = x
        valid = True
        if s in sums:
            valid = False
        else:
            for a in reversed(seq):
                s += a
                if s in sums:
                    valid = False
                    break
        if valid:
            seq.append(x)
            s = x
            sums.add(s)
            for a in reversed(seq[:-1]):
                s += a
                sums.add(s)
    return seq

def _dfs_extend(current_seq, current_sums, min_val, best_len, best_seq, deadline):
    """
    Depth First Search to extend the sequence.
    Tries candidates in descending order (largest first).
    """
    nonlocal best_len, best_seq
    
    if len(current_seq) > best_len:
        best_len = len(current_seq)
        best_seq = list(current_seq)
        
    # Pruning
    if len(current_seq) + (256 - min_val + 1) <= best_len:
        return

    if time.time() > deadline:
        return

    last_val = current_seq[-1] if current_seq else 0
    
    for x in range(256, min_val - 1, -1):
        if time.time() > deadline:
            return
        
        if x <= last_val:
            continue
            
        s = x
        valid = True
        if s in current_sums:
            valid = False
        else:
            for a in reversed(current_seq):
                s += a
                if s in current_sums:
                    valid = False
                    break
        
        if valid:
            new_sums_to_add = []
            s = x
            new_sums_to_add.append(s)
            for a in reversed(current_seq):
                s += a
                new_sums_to_add.append(s)
                
            current_seq.append(x)
            for ns in new_sums_to_add:
                current_sums.add(ns)
                
            _dfs_extend(current_seq, current_sums, x + 1, best_len, best_seq, deadline)
            
            current_seq.pop()
            for ns in new_sums_to_add:
                current_sums.remove(ns)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_seq = _greedy_largest()
    best_len = len(best_seq)
    
    current_seq = []
    current_sums = set()
    
    _dfs_extend(current_seq, current_sums, 1, best_len, best_seq, deadline)
    
    return best_seq
