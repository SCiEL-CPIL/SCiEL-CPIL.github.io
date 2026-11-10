"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    n = 512
    start_time = time.time()
    time_limit = 85  # Safety margin
    
    def is_valid_insertion(x, selected):
        # Check if adding x to selected creates a 4-AP
        # x can be part of APs:
        # 1. x is 1st: x, x+d, x+2d, x+3d
        # 2. x is 2nd: x-3d, x-2d, x-d, x  (Wait, x-3d, x-2d, x-d are in selected)
        # 3. x is 3rd: x-2d, x-d, x, x+d   (x-2d, x-d, x+d in selected)
        # 4. x is 4th: x-3d, x-2d, x-d, x   (x-3d, x-2d, x-d in selected)
        
        # We just need to check if there exists d such that the other 3 terms are in selected.
        
        # Case 1: x, x+d, x+2d, x+3d
        # d >= 1, x+3d <= n
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in selected and (x + 2 * d) in selected and (x + 3 * d) in selected:
                return False
        
        # Case 2: x-3d, x-2d, x-d, x
        # x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            if (x - d) in selected and (x - 2 * d) in selected and (x - 3 * d) in selected:
                return False
        
        # Case 3: x-2d, x-d, x, x+d
        # x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2
        # x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - d) in selected and (x - 2 * d) in selected and (x + d) in selected:
                return False
        
        # Case 4: x-d, x, x+d, x+2d
        # x-d >= 1 => d <= x-1
        # x+2d <= n => 2d <= n-x => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in selected and (x + d) in selected and (x + 2 * d) in selected:
                return False
                
        return True

    best_set = set()
    best_size = 0
    
    # 1. Standard Greedy
    selected = set()
    for x in range(1, n + 1):
        if is_valid_insertion(x, selected):
            selected.add(x)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
        
    # 2. Randomized Greedy with multiple restarts
    candidates = list(range(1, n + 1))
    num_restarts = 0
    while time.time() - start_time < time_limit * 0.5: # Use half the time for random restarts
        random.shuffle(candidates)
        selected = set()
        for x in candidates:
            if is_valid_insertion(x, selected):
                selected.add(x)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
        num_restarts += 1
        
    # 3. Local Search / Expansion on the best set found
    # Try to add elements not in best_set in a random order
    # If we can't add directly, maybe we can swap?
    # For now, just try to add any remaining element that fits.
    # We do this multiple times with different orderings to find different local optima.
    
    remaining = [x for x in range(1, n + 1) if x not in best_set]
    
    while time.time() - start_time < time_limit * 0.9:
        random.shuffle(remaining)
        current_set = set(best_set)
        added_any = False
        for x in remaining:
            if is_valid_insertion(x, current_set):
                current_set.add(x)
                added_any = True
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            remaining = [x for x in range(1, n + 1) if x not in best_set]
        if not added_any:
            break # No more elements can be added in this configuration
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
