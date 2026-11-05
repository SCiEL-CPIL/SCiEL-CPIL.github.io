import sys

def construct():
    """
    Returns a strictly increasing sequence of integers in [1, 256]
    such that sums of all nonempty consecutive blocks are distinct.
    Maximizes the sequence length.
    """
    sys.setrecursionlimit(10000)
    
    best_sequence = []
    global_best_len = 0
    gaps = []
    used_sums = set()
    
    def dfs(last_gap):
        nonlocal best_sequence, global_best_len
        
        current_len = len(gaps)
        
        # Pruning: Max possible additional elements
        # The smallest next gap is last_gap + 1.
        # The largest possible gap is 256.
        # So max additional elements is 256 - last_gap.
        if current_len + (256 - last_gap) <= global_best_len:
            return
            
        # Try all possible next gaps
        for g in range(last_gap + 1, 257):
            # Calculate new consecutive block sums ending at g
            new_sums = []
            s = g
            new_sums.append(s)
            # Add suffix sums of existing gaps
            for i in range(len(gaps) - 1, -1, -1):
                s += gaps[i]
                new_sums.append(s)
                
            # Check for collisions with existing sums
            collision = False
            for ns in new_sums:
                if ns in used_sums:
                    collision = True
                    break
            
            if not collision:
                # Commit
                gaps.append(g)
                used_sums.update(new_sums)
                
                # Update best solution found so far
                if current_len + 1 > global_best_len:
                    global_best_len = current_len + 1
                    best_sequence = list(gaps)
                    
                # Recurse
                dfs(g)
                
                # Backtrack
                gaps.pop()
                for ns in new_sums:
                    used_sums.remove(ns)
                    
    dfs(0)
    
    return best_sequence
