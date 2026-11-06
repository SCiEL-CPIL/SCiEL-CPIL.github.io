import time
import sys

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

# Max possible sum is 284 + 284 = 568
# We need an array of size 569 (indices 0 to 568)
SUMS_SIZE = 570

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    # Initial prescribed set
    initial_set = [1, 2, 5, 11]
    initial_sums = bytearray(SUMS_SIZE)
    
    # Mark initial sums
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_sums[s] = 1
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    # Candidates are 12 to 284
    # We filter out candidates that conflict with the initial set immediately
    candidates = []
    for x in range(12, 285):
        valid = True
        # Check 2*x
        if 2 * x < SUMS_SIZE and initial_sums[2 * x]:
            valid = False
        # Check x + y for y in initial_set
        if valid:
            for y in initial_set:
                s = x + y
                if s < SUMS_SIZE and initial_sums[s]:
                    valid = False
                    break
        if valid:
            candidates.append(x)
            
    # Sort candidates. 
    # Heuristic: Descending order often works well for Golomb rulers to spread out marks.
    # Or ascending? Let's try descending first.
    candidates.sort(reverse=True)
    
    # We will use a recursive DFS with pruning.
    # To avoid recursion depth issues and allow global time checks, we use a helper.
    
    # We try to find a solution of size 20, then 19, etc.
    # Since 20 is the theoretical max for length 283/284, we start there.
    # If we find one, we return immediately.
    
    # State: current_set (list), current_sums (bytearray), start_idx (int in candidates)
    
    def dfs(current_set, current_sums, start_idx, target_size):
        global best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: If we can't reach target_size with remaining candidates
        if current_size + (len(candidates) - start_idx) < target_size:
            return False
            
        # Iterate through candidates from start_idx
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Check if x can be added
            # 1. 2*x
            if 2 * x < SUMS_SIZE and current_sums[2 * x]:
                continue
                
            # 2. x + y for y in current_set
            can_add = True
            for y in current_set:
                s = x + y
                if s < SUMS_SIZE and current_sums[s]:
                    can_add = False
                    break
                    
            if not can_add:
                continue
                
            # Add x
            current_set.append(x)
            current_sums[2 * x] = 1
            for y in current_set:
                if y != x:
                    current_sums[x + y] = 1
            
            if dfs(current_set, current_sums, i + 1, target_size):
                return True
                
            # Backtrack
            current_set.pop()
            current_sums[2 * x] = 0
            for y in current_set:
                current_sums[x + y] = 0
                
        return False

    # Try sizes from 20 down to best_size + 1
    # We know best_size is at least 4.
    # Max size is 20.
    for target in range(20, best_size, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Reset state for new target? 
        # No, we can just run DFS for this target.
        # We want to find ANY set of this size.
        # We reuse the initial state.
        
        # Note: dfs modifies current_set and current_sums. We need fresh copies.
        # But passing copies is expensive. 
        # We can use the fact that dfs backtracks fully if it returns False.
        # However, if it returns True, we are done.
        
        # One issue: If dfs returns False, it has backtracked to the initial state.
        # So we can reuse the same initial structures?
        # Yes, if we ensure dfs backtracks all changes.
        
        # Let's optimize: Pass the initial structures directly?
        # No, because `current_set` is modified. 
        # We can pass copies of the *initial* set and sums for each target?
        # Or just once, and rely on backtracking?
        # If dfs(target) fails, it backtracks to the state it was called with.
        # So if we call dfs(initial_set, initial_sums, 0, target), and it fails,
        # initial_set and initial_sums should be restored to their original state.
        
        # Let's verify the backtracking logic.
        # Yes, `current_set.pop()` and `current_sums[...] = 0` are called.
        
        # So we can call dfs repeatedly with the same initial objects?
        # Yes, provided the first call backtracks completely.
        
        # However, there is a risk: if the time limit is hit, we return False 
        # but might not have backtracked completely?
        # The time check is at the top. If we hit it, we return False immediately.
        # This might leave the state dirty!
        
        # To be safe, let's create fresh copies for each target attempt, or 
        # ensure we only run one target at a time and restart if dirty.
        # Given the time limit, creating copies of a small list/bytearray is cheap.
        
        init_set_copy = initial_set[:]
        init_sums_copy = initial_sums[:]
        
        if dfs(init_set_copy, init_sums_copy, 0, target):
            return best_set
            
    return best_set
