import time

def construct():
    best_seq = []
    start_time = time.time()
    # Use a deadline slightly before the 60s limit to be safe
    deadline = start_time + 55.0
    
    # State variables
    seq = []
    prefix = [0]
    used_sums = set()
    
    def dfs():
        nonlocal best_seq
        
        # Check time
        if time.time() > deadline:
            return
            
        # Update best if current is better
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # If we have a very long sequence, we might consider stopping?
        # But we want max length.
        
        # Try adding next element
        # Iterate x from 1 to 256
        # Heuristic: Smaller x first?
        for x in range(1, 257):
            if time.time() > deadline:
                return
                
            new_p = prefix[-1] + x
            
            # Check for collisions
            # New sums are new_p - p for p in prefix
            valid = True
            new_sums = []
            for p in prefix:
                s = new_p - p
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
                
            if valid:
                # Apply change
                seq.append(x)
                prefix.append(new_p)
                used_sums.update(new_sums)
                
                # Recurse
                dfs()
                
                # Backtrack
                seq.pop()
                prefix.pop()
                used_sums.difference_update(new_sums)
                
    dfs()
    
    return best_seq
