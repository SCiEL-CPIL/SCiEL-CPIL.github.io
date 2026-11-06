import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initial greedy solution as fallback
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
    best_len = len(values)
    
    # Try to find longer sequences using backtracking
    # We try lengths from 15 down to best_len + 1
    for target_len in range(15, best_len, -1):
        if time.time() > deadline:
            break
            
        # Randomized backtracking
        found = _search_for_length(target_len, deadline)
        if found:
            best_seq = found
            best_len = target_len
            # If we found a very long one, we can stop or try to go higher
            # But the loop goes down, so we should go up?
            # Actually, we want the MAX length.
            # So we should try 20, 19, 18...
            # If we find 10, we don't need to try 9.
            # So loop should be 20 down to best_len + 1.
            
    return best_seq

def _search_for_length(target_len, deadline):
    # Heuristic: Start with small numbers to leave room? 
    # Or use random restarts.
    # Let's try a few random restarts.
    
    # To make backtracking efficient, we can restrict the search space.
    # If we pick x1, x2...
    # We can use a recursive function.
    
    def backtrack(seq, used_sums, last_val):
        if time.time() > deadline:
            return None
        
        if len(seq) == target_len:
            return seq
        
        # Pruning: Not enough numbers left
        # Max possible length from last_val+1 to 256 is 256 - last_val
        if len(seq) + (256 - last_val) < target_len:
            return None
            
        # Determine next value range
        # To find a solution faster, we can randomize the start or step
        # But simple iteration might work if we pick "good" candidates.
        
        # Strategy: Try candidates in a randomized order or with a step
        # To avoid deep recursion on bad paths, we can limit the depth or use a heuristic.
        
        # Let's try candidates from last_val + 1 to 256
        # But 256 is small.
        
        # Optimization: If we are close to target_len, we must pick small numbers.
        # If we have k slots left, we need k numbers in [last_val+1, 256].
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return None
                
            # Check if x can be added
            # Calculate new sums
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                
            if used_sums.intersection(new_sums):
                continue
                
            # Recurse
            seq.append(x)
            new_used = used_sums.union(new_sums)
            result = backtrack(seq, new_used, x)
            if result:
                return result
            seq.pop()
            
        return None

    # Run a few random restarts with different initial choices or random shuffles of candidate order?
    # Actually, the order 1..256 is deterministic.
    # To find diverse solutions, we can randomize the "start" or "step".
    # Or just run the deterministic backtrack, but it might get stuck in a local optimum of "no solution" for that path.
    # Since the space is small, let's try a few random permutations of the "priority" of numbers?
    # Simpler: Just run the deterministic backtrack. If it fails, try again with a slight modification?
    # The problem is that backtracking might explore a huge tree.
    
    # Better heuristic: 
    # To fit L numbers in 1..256, the average gap is 256/L.
    # We can guide the search by preferring numbers that are roughly spaced.
    
    # Let's try a randomized local search approach instead, which is often faster for these "satisfiability" like problems.
    # 1. Generate a random increasing sequence of length target_len from 1..256.
    # 2. Check if valid.
    # 3. If not, try to fix collisions by swapping/adjusting elements?
    # This is hard to define "fix".
    
    # Let's stick to backtracking but add a "random shuffle" of the candidate list at each step?
    # That would be expensive.
    
    # Alternative: 
    # The sequence 1, 3, 7, 15... worked well.
    # Let's try to generate sequences based on geometric progressions.
    
    # For now, the deterministic backtrack with pruning might be enough for L=10,11.
    # Let's run it.
    
    result = backtrack([], set(), 0)
    return result
