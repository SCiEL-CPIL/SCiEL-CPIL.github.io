import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    best_seq, best_sums = greedy()
    best_len = len(best_seq)
    
    seq = []
    sums = set()
    last_val = 0
    
    def dfs():
        nonlocal best_seq, best_sums, best_len
        
        if time.time() > deadline:
            return False # Timeout
        
        if len(seq) > best_len:
            best_seq = seq.copy()
            best_sums = sums.copy()
            best_len = len(seq)
            
        # Pruning
        if len(seq) + (256 - last_val) <= best_len:
            return True
        
        if last_val == 256:
            return True
            
        for x in range(last_val + 1, 257):
            # Check validity
            total = x
            if total in sums:
                continue
            
            new_sums = [total]
            valid = True
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                if total in sums:
                    valid = False
                    break
            
            if valid:
                seq.append(x)
                sums.update(new_sums)
                old_last = last_val
                last_val = x
                
                # Recurse
                # We continue searching even if we found a better solution, 
                # because we might find an even better one.
                # But if we hit timeout, we stop.
                if not dfs():
                    # Undo before returning
                    seq.pop()
                    for s in new_sums:
                        sums.remove(s)
                    last_val = old_last
                    return False
                    
                # Undo
                seq.pop()
                for s in new_sums:
                    sums.remove(s)
                last_val = old_last
                
                if time.time() > deadline:
                    return False
                    
        return True

    dfs()
    
    return best_seq
