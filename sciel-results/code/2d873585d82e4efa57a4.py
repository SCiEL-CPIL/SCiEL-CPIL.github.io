import sys
import time

def construct():
    sys.setrecursionlimit(10000)
    start_time = time.time()
    time_limit = 55  # Leave 5s for overhead
    
    best_len = 0
    best_seq = []
    
    # Greedy baseline
    greedy_seq = []
    greedy_sums = set()
    for x in range(1, 257):
        new_sums = []
        s = 0
        valid = True
        s = x
        if s in greedy_sums:
            valid = False
        else:
            new_sums.append(s)
            for a in reversed(greedy_seq):
                s += a
                if s in greedy_sums:
                    valid = False
                    break
                new_sums.append(s)
        if valid:
            greedy_seq.append(x)
            greedy_sums.update(new_sums)
            
    if len(greedy_seq) > best_len:
        best_len = len(greedy_seq)
        best_seq = greedy_seq[:]

    current_seq = []
    current_sums = set()
    
    def dfs():
        nonlocal best_len, best_seq
        
        if time.time() - start_time > time_limit:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        if len(current_seq) == 256:
            return
            
        last_val = current_seq[-1] if current_seq else 0
        
        if len(current_seq) + (256 - last_val) <= best_len:
            return
            
        for x in range(last_val + 1, 257):
            new_sums = []
            s = x
            valid = True
            if s in current_sums:
                valid = False
            else:
                new_sums.append(s)
                for a in reversed(current_seq):
                    s += a
                    if s in current_sums:
                        valid = False
                        break
                    new_sums.append(s)
                    
            if valid:
                for ns in new_sums:
                    current_sums.add(ns)
                current_seq.append(x)
                
                dfs()
                
                current_seq.pop()
                for ns in new_sums:
                    current_sums.discard(ns)

    dfs()
    
    return best_seq
