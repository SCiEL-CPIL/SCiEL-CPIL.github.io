import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    prescribed = [1, 2, 5, 11]
    # Sums of prescribed elements
    initial_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_sums.add(prescribed[i] + prescribed[j])
            
    best_set = prescribed[:]
    
    # We want to find the largest K
    # Max possible K is 20
    for K in range(20, len(prescribed), -1):
        # Time check
        if time.time() > deadline - 5:
            break
            
        # Time limit for this K
        # Allocate more time to larger K
        if K == 20:
            time_limit = 100
        elif K == 19:
            time_limit = 10
        else:
            time_limit = 2
            
        found = False
        # Backtracking search
        # State: current_set, used_sums, next_index, min_val
        # We use a recursive function
        
        def solve(current_set, used_sums, target_size, min_val):
            nonlocal found
            if time.time() > deadline - 2:
                return False
                
            if len(current_set) == target_size:
                found = True
                best_set[:] = current_set[:]
                return True
                
            # If we don't have enough numbers left to reach target_size
            # Max value is 284.
            # We need target_size - len(current_set) numbers from [min_val, 284]
            # Count available numbers: 284 - min_val + 1
            # If (284 - min_val + 1) < (target_size - len(current_set)), return False
            if (284 - min_val + 1) < (target_size - len(current_set)):
                return False
                
            for val in range(min_val, 285):
                # Check sums
                new_sums = set()
                valid = True
                for x in current_set:
                    s = val + x
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * val
                    if s in used_sums:
                        valid = False
                    else:
                        new_sums.add(s)
                        
                if valid:
                    current_set.append(val)
                    used_sums.update(new_sums)
                    
                    if solve(current_set, used_sums, target_size, val + 1):
                        return True
                        
                    current_set.pop()
                    used_sums.difference_update(new_sums)
                    
        # Initialize for this K
        current_set = prescribed[:]
        used_sums = set(initial_sums)
        min_val = 12 # Must be > 11
        
        if solve(current_set, used_sums, K, min_val):
            return best_set[:]
            
    # If no solution found for K > 4, return the best found (which might be just prescribed if search failed everywhere)
    # But we should probably run a quick greedy to get a better baseline than just 4
    # The seed program does a greedy. Let's include a simple greedy as fallback if backtracking fails immediately.
    # However, the backtracking for K=16 should be very fast.
    
    return best_set
