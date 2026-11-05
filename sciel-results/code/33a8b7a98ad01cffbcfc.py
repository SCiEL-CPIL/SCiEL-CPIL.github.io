import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    prescribed = [1, 2, 5, 11]
    # Initialize sums
    sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            sums.add(prescribed[i] + prescribed[j])
            
    # We want to find a set of size 20
    target_size = 20
    current_set = prescribed[:]
    
    def is_valid(candidate, current_set, sums):
        # Check if adding candidate creates duplicate sums
        # New sums: candidate + y for y in current_set, and candidate + candidate
        for y in current_set:
            s = candidate + y
            if s in sums:
                return False
        s_2 = candidate + candidate
        if s_2 in sums:
            return False
        return True

    def get_new_sums(candidate, current_set):
        new_sums = set()
        for y in current_set:
            new_sums.add(candidate + y)
        new_sums.add(candidate + candidate)
        return new_sums

    def backtrack(idx, current_set, sums):
        if time.time() > deadline:
            return current_set
        
        if len(current_set) == target_size:
            return current_set
        
        # If we can't reach target size with remaining candidates, prune
        # Max possible additions = 284 - current_set[-1] (roughly, but candidates are specific)
        # Actually, we iterate candidates in order.
        
        # Candidates are integers from current_set[-1] + 1 to 284
        # But we can skip some? No, we must consider all.
        
        start_val = current_set[-1] + 1
        if start_val > 284:
            return current_set
            
        # Heuristic: Try candidates in a specific order?
        # Randomization might help avoid local minima if backtracking is slow.
        # But let's try deterministic first.
        
        # To speed up, we can pre-filter candidates?
        # No, just iterate.
        
        for x in range(start_val, 285):
            if time.time() > deadline:
                return current_set
                
            if is_valid(x, current_set, sums):
                new_sums = get_new_sums(x, current_set)
                new_set = current_set + [x]
                
                # Recursive call
                result = backtrack(x, new_set, sums | new_sums)
                if len(result) == target_size:
                    return result
                
        return current_set

    # Initial call
    # We can also try to randomize the order of candidates to explore different branches
    # But for now, let's stick to ordered.
    
    # Optimization: If deterministic is too slow, we can add random restarts.
    # Let's add a simple loop with random shuffling of candidates if we don't find it quickly.
    
    # However, the search space is small.
    # 16 more elements to choose from ~270 candidates.
    # Combinations are huge, but pruning is strong.
    
    # Let's try to run the backtracking.
    # If it times out, we return the best found (which is the current_set at the time of timeout, 
    # but backtrack returns the deepest valid set found? No, it returns current_set if no solution found).
    
    # To ensure we return a valid set, we should track the best (largest) valid set found so far.
    
    best_set = prescribed[:]
    
    def search_with_tracking(idx, current_set, sums):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(current_set) == target_size:
            return
            
        start_val = current_set[-1] + 1
        if start_val > 284:
            return
            
        # Try to add elements
        # Heuristic: Try smaller gaps first? Or larger?
        # Let's just iterate.
        
        for x in range(start_val, 285):
            if time.time() > deadline:
                return
                
            if is_valid(x, current_set, sums):
                new_sums = get_new_sums(x, current_set)
                new_set = current_set + [x]
                
                search_with_tracking(x, new_set, sums | new_sums)
                
                if len(best_set) == target_size:
                    return

    search_with_tracking(0, prescribed, sums)
    
    return best_set
