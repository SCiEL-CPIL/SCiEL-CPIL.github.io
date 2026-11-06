import time
import sys

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Initial sums from prescribed elements
    initial_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_sums.add(prescribed[i] + prescribed[j])
            
    # We want to maximize the size.
    # The max possible size is around 20.
    # We can try to find a solution of size 20, then 19, etc.
    # Or just run a depth-first search that tries to add as many as possible.
    # Given the time limit, a simple DFS with good heuristics might find the optimal quickly.
    # Let's try to find the largest possible set.
    
    best_size = len(prescribed)
    best_solution = prescribed[:]
    
    # To speed up, we can pre-compute potential candidates or just iterate 1..284.
    # Heuristic: Try numbers in a specific order. 
    # Often, spreading out numbers helps. 
    # However, for Sidon sets, there are known constructions.
    # A simple heuristic is to try numbers in ascending order, but skip those that conflict.
    # To improve on greedy, backtracking allows us to "undo" bad choices.
    
    def backtrack(current_set, used_sums, current_size, target_size):
        nonlocal best_size, best_solution
        
        if current_size > best_size:
            best_size = current_size
            best_solution = current_set[:]
            
        # If we reached the target size, we can stop this branch
        if current_size == target_size:
            return True
            
        # Time check
        if time.time() - start_time > time_limit:
            return False
            
        last_element = current_set[-1] if current_set else 0
        
        # Try adding new elements
        # Optimization: We can restrict the search space.
        # If we are looking for a set of size K, the numbers should be spread out.
        # A simple range is 1..284.
        # To make it faster, we can iterate candidates in a smart order.
        # For example, if we just added a large number, the next one should be larger.
        # So we only consider candidates > last_element.
        
        for x in range(last_element + 1, 285):
            if x in current_set: # Should be handled by range, but safe
                continue
                
            # Check if x can be added
            valid = True
            new_sums = set()
            
            # Check 2*x
            if 2 * x in used_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                
                # Check x + y for all y in current_set
                for y in current_set:
                    s = x + y
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                    
            if valid:
                # Add x
                current_set.append(x)
                used_sums.update(new_sums)
                
                # Recurse
                if backtrack(current_set, used_sums, current_size + 1, target_size):
                    return True
                    
                # Backtrack
                current_set.pop()
                for s in new_sums:
                    used_sums.discard(s)
                    
        return False

    # We will try to find a solution of size 20, then 19, etc.
    # Start from the highest likely size.
    # The max size for n=284 is 20.
    # We start with the prescribed elements.
    
    # Sort prescribed elements to start the recursion correctly
    current_set = sorted(prescribed)
    used_sums = initial_sums
    
    # Try target sizes from 20 down to 4
    # Note: The prescribed elements are 4.
    for target_size in range(20, 3, -1):
        # Reset state for each target size attempt? 
        # Actually, we can just run the search. If it finds a solution of size 'target_size', we record it.
        # The backtrack function updates best_solution.
        # We can stop if we find a solution of size 20, as that is the theoretical max.
        
        # We need to be careful: the search is depth-first.
        # It will try to extend the current set.
        # If it fails to reach target_size, it returns False.
        
        # To avoid re-exploring the same partial sets for different targets, 
        # we could just run one search that tries to maximize depth.
        # However, the previous implementation overwrote best_solution only if size increased.
        # Let's modify the approach: Run a single search that tries to go as deep as possible.
        # But standard DFS might get stuck in a deep but small branch.
        # Iterative Deepening or simply trying to reach high targets is better.
        
        # Let's try to reach 'target_size'.
        # If successful, we have a solution of that size.
        # Since we iterate 20, 19, ..., the first success is the best.
        
        # We need to reset the search state? 
        # The `backtrack` function modifies `current_set` and `used_sums` in place and restores them.
        # So we can call it with the initial state.
        
        if backtrack(current_set, used_sums, len(current_set), target_size):
            # Found a solution of this size
            # Since we are going downwards, this is the maximum found so far.
            # If we found 20, we can break.
            if target_size == 20:
                break
            # Otherwise, we continue to see if we can find a larger one? 
            # No, we are iterating 20 down. So if we found 19, we already tried 20 and failed.
            # So we can break? 
            # Wait, if we tried 20 and failed, then 19 is the max.
            # So we can break after the first success.
            break
            
    # If no specific target was hit (e.g. if search timed out before finishing a target),
    # best_solution holds the best partial solution found during the attempts.
    # However, the `backtrack` function updates `best_solution` whenever it finds a *larger* set,
    # even if it doesn't reach the target.
    # So `best_solution` should be valid.
    
    return best_solution
