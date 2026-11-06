import time
import sys
import random

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

SUMS_SIZE = 570 # Max sum 568

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    initial_set = [1, 2, 5, 11]
    initial_sums_mask = 0
    
    # Mark initial sums
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_sums_mask |= (1 << s)
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    # Candidates
    candidates = []
    for x in range(12, 285):
        # Check conflict with initial_set
        valid = True
        if initial_sums_mask & (1 << (2*x)):
            valid = False
        if valid:
            for y in initial_set:
                if initial_sums_mask & (1 << (x+y)):
                    valid = False
                    break
        if valid:
            candidates.append(x)
            
    # Heuristic sort: Descending
    candidates.sort(reverse=True)
    
    # DFS with Bitset
    def dfs(current_set, sums_mask, start_idx, target_size):
        global best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning
        if current_size + (len(candidates) - start_idx) < target_size:
            return False
            
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Check conflict
            # 1. 2*x
            if sums_mask & (1 << (2*x)):
                continue
                
            # 2. x + y
            required_mask = 1 << (2*x)
            can_add = True
            for y in current_set:
                bit = 1 << (x + y)
                if sums_mask & bit:
                    can_add = False
                    break
                required_mask |= bit
                
            if not can_add:
                continue
                
            # Add
            current_set.append(x)
            new_mask = sums_mask | required_mask
            
            if dfs(current_set, new_mask, i + 1, target_size):
                return True
                
            # Backtrack
            current_set.pop()
            
        return False

    # Try sizes from 20 down to best_size + 1
    for target in range(20, best_size, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try multiple random restarts for this target
        # We limit restarts to avoid wasting time if the target is impossible
        max_restarts = 50 
        for r in range(max_restarts):
            if time.time() - start_time > time_limit - 2:
                break
                
            # Shuffle candidates
            # We need to be careful: shuffling the global list affects all calls.
            # So we create a shuffled copy or shuffle in place and restore?
            # Shuffling a list of ~270 items is fast.
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            
            # We need to pass the shuffled list to dfs.
            # But dfs uses `candidates` global.
            # Let's modify dfs to take the candidate list as an argument or use a closure variable that is updated.
            # Easier: pass `cand_list` to dfs.
            
            # Let's refactor dfs to accept cand_list
            # ... (rewrite dfs signature)
            
            pass # Placeholder for logic below

    return best_set
