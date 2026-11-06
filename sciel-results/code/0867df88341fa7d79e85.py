import time
import sys

def construct():
    sys.setrecursionlimit(10000)
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe within 90s

    def is_time_up():
        return time.time() > deadline

    # 1. Greedy Baseline
    def greedy():
        values = []
        sums_mask = 0
        tails = [] # list of sums of suffixes
        
        for x in range(1, 257):
            new_tails = [x]
            for t in tails:
                new_tails.append(x + t)
            
            # Check collisions using bitmask
            new_mask = 0
            for s in new_tails:
                new_mask |= (1 << s)
                
            if sums_mask & new_mask:
                continue
            
            # Accept
            values.append(x)
            sums_mask |= new_mask
            tails = new_tails
            
        return values, sums_mask, tails

    base_seq, base_sums, base_tails = greedy()
    best_seq = base_seq
    base_len = len(base_seq)
    
    # 2. Backtracking Search with Bitmasks
    def dfs(seq, sums_mask, tails, last, target_len, current_len):
        if is_time_up():
            return None, "timeout"
        
        if current_len == target_len:
            return seq[:], "success"
        
        # Pruning: Not enough numbers left
        if 256 - last < target_len - current_len:
            return None, "pruned"
            
        for x in range(last + 1, 257):
            # Calculate new tails
            new_tails = [x]
            for t in tails:
                new_tails.append(x + t)
            
            # Create mask for new_tails
            new_mask = 0
            for s in new_tails:
                new_mask |= (1 << s)
                
            # Check collision
            if sums_mask & new_mask:
                continue
            
            # Update state
            new_sums_mask = sums_mask | new_mask
            new_seq = seq + [x]
            
            result, status = dfs(new_seq, new_sums_mask, new_tails, x, target_len, current_len + 1)
            
            if status == "success":
                return result, "success"
            elif status == "timeout":
                return None, "timeout"
                
        return None, "exhausted"

    # Try to find longer sequences
    current_target = base_len + 1
    
    while current_target <= 256:
        if is_time_up():
            break
            
        result, status = dfs([], 0, [], 0, current_target, 0)
        
        if status == "success":
            best_seq = result
            current_target += 1
        elif status == "exhausted":
            # No sequence of this length exists
            break
        else: # timeout
            break
            
    return best_seq
