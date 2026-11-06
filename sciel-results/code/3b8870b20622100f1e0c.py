import sys

def construct():
    best_seq = []
    best_len = 0
    
    def get_new_sums(seq, x):
        """Compute all consecutive block sums ending at x when x is appended to seq"""
        sums = [x]
        running = x
        for a in reversed(seq):
            running += a
            sums.append(running)
        return sums
    
    def backtrack(seq, sums_set):
        nonlocal best_seq, best_len
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        if not seq:
            start = 1
        else:
            start = seq[-1] + 1
        
        # Pruning: can't beat best even with all remaining numbers
        if len(seq) + (256 - start + 1) <= best_len:
            return
        
        for x in range(start, 257):
            new_sums = get_new_sums(seq, x)
            
            # Forward checking: reject if any new sum collides with existing
            if any(s in sums_set for s in new_sums):
                continue
            
            new_sums_set = set(new_sums)
            backtrack(seq + [x], sums_set | new_sums_set)
    
    backtrack([], set())
    return best_seq
