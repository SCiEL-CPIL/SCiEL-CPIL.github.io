import time
import sys

# Increase recursion limit if necessary
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # Greedy baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_values = values[:]
    best_len = len(values)
    
    # DFS to find longer sequences
    # We try to find a sequence of length L
    # We start from L = best_len + 1
    
    def dfs(current_vals, current_sums, last_val, remaining_len):
        nonlocal best_values, best_len, deadline
        
        if time.time() > deadline:
            return False
        
        if remaining_len == 0:
            if len(current_vals) > best_len:
                best_len = len(current_vals)
                best_values = current_vals[:]
            return True
        
        # Pruning: Not enough numbers left to pick
        if 256 - last_val < remaining_len:
            return False
            
        # Try to pick the next value
        # Heuristic: Try values that are likely to be "compatible"
        # We iterate from last_val + 1 to 256
        # To potentially find solutions faster, we might want to try
        # values that create large sums (larger x) or small sums (smaller x)?
        # Let's try ascending order first.
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return False
            
            # Calculate new tails
            tails, total = [x], x
            for a in reversed(current_vals):
                total += a
                tails.append(total)
                
            # Check for collisions
            if current_sums.intersection(tails):
                continue
                
            # No collision, recurse
            new_sums = current_sums.union(tails)
            # Optimization: copy sums only if necessary? 
            # Sets are immutable in usage here? No, we pass a new set.
            # Creating a new set of size ~100 is fast.
            
            if dfs(current_vals + [x], new_sums, x, remaining_len - 1):
                return True
                
        return False

    # Try to find longer sequences
    # We can try increasing L
    # Start from best_len + 1
    # If we find one, we can try to extend it further?
    # Or just restart DFS for L+1?
    
    # Since DFS for L might be hard, we can also try to extend the current best
    # by backtracking from the end?
    
    # Let's try a simple loop for L
    # But DFS for L=12 might be slow.
    # We can use a beam search or randomized restarts.
    
    # For now, let's just run DFS for L = best_len + 1
    # If successful, try best_len + 2, etc.
    
    # To make it faster, we can randomize the order of x?
    # No, x must be > last_val.
    
    # Let's try a different heuristic:
    # Try to pick x such that the new sums are as "spread out" as possible?
    # Hard to measure.
    
    # Let's just run DFS.
    
    current_L = best_len
    while time.time() < deadline - 5.0:
        target_L = current_L + 1
        # We need to search for a sequence of length target_L
        # We can restart the DFS from the beginning for this target length
        # Or we can try to extend the best found so far?
        # Extending is faster if we use the existing prefix.
        
        # Let's try to extend best_values
        # But best_values might be a dead end for length target_L.
        # So we should do a full DFS for length target_L.
        
        # To speed up full DFS, we can use pruning.
        # Also, we can try multiple random starting points?
        # No, we must start from scratch.
        
        # Let's run DFS for target_L
        # We need a fresh state.
        
        # To avoid exponential blowup, we limit the depth or use a heuristic.
        # Since target_L is small (maybe < 20), it should be okay.
        
        found = dfs([], set(), 0, target_L)
        if found:
            current_L = target_L
        else:
            # If we didn't find a sequence of length target_L, 
            # maybe we can find a different sequence of length current_L 
            # that allows extension?
            # For simplicity, we stop if we fail to find target_L.
            # Or we can try random restarts.
            break
            
    return best_values

# Test locally? No, just output.
