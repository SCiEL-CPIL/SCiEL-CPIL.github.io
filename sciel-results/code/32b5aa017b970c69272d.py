import time

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
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
    
    # Get greedy baseline
    base_seq, base_sums = greedy()
    best = base_seq
    
    # Backtracking to extend beyond greedy
    def backtrack(seq, sums, depth):
        nonlocal best
        if time.time() > deadline:
            return
        
        if len(seq) > len(best):
            best = list(seq)
        
        if len(seq) >= 16:  # upper bound heuristic
            return
        
        last = seq[-1] if seq else 0
        # Try candidates from last+1 to 256
        # Heuristic: try smaller first to keep values low, but also try some larger
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            
            # Compute new tails
            tails = []
            total = x
            tails.append(total)
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            # Check if any tail is in sums
            if any(t in sums for t in tails):
                continue
            
            # Valid - recurse
            seq.append(x)
            sums.update(tails)
            backtrack(seq, sums, depth + 1)
            sums.difference_update(tails)
            seq.pop()
    
    # Try to extend the greedy solution
    backtrack(list(base_seq), set(base_sums), len(base_seq))
    
    return best
