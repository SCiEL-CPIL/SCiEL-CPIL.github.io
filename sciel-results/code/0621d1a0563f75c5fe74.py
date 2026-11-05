import time
import random

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses optimized backtracking with bitmasks for speed.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (evaluator kills at 150s)
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    # Initialize chosen mask and used sums mask
    chosen_mask = 0
    for r in required:
        chosen_mask |= (1 << r)
        
    used_sums_mask = 0
    for i in range(len(required)):
        for j in range(i, len(required)):
            s = required[i] + required[j]
            used_sums_mask |= (1 << s)
            
    best_solution = list(required)
    best_size = len(best_solution)
    
    # Pool of all other candidates
    pool_base = [x for x in range(1, max_val + 1) if x not in required]
    
    def dfs(chosen_mask, used_sums_mask, current_size, remaining_pool):
        nonlocal best_solution, best_size
        
        # Update best if we found a larger set
        if current_size > best_size:
            # Extract elements from mask
            sol = []
            m = chosen_mask
            while m:
                lsb = m & -m
                sol.append(lsb.bit_length() - 1)
                m ^= lsb
            best_solution = sol
            best_size = current_size
            
        # Time check
        if time.time() > deadline:
            return False
            
        # If no candidates left, stop
        if not remaining_pool:
            return False
            
        # Heuristic: if we can't beat best size even with all remaining, stop
        if current_size + len(remaining_pool) <= best_size:
            return False
            
        # Try each candidate in the remaining pool
        for i in range(len(remaining_pool)):
            x = remaining_pool[i]
            
            # Check for sum collisions
            # New sums introduced by x:
            # 1. 2*x
            # 2. x + y for all y in chosen_mask
            # (chosen_mask << x) creates bits at positions y+x for all y in chosen_mask
            # We need to check if any of these positions are already in used_sums_mask
            # Also check if 2*x is in used_sums_mask
            
            # Bitmask of new sums
            new_sums = (1 << (2 * x)) | (chosen_mask << x)
            
            # If any new sum is already used, skip x
            if (used_sums_mask & new_sums) != 0:
                continue
                
            # Valid move: add x
            new_chosen = chosen_mask | (1 << x)
            new_used = used_sums_mask | new_sums
            
            # Recurse with remaining candidates after i
            # We pass a slice to avoid modifying the original list, but slicing is expensive.
            # Alternative: pass index and original list, or use a list of indices.
            # For performance, slicing a small list (max 280 items) is okay, but 
            # passing the index is faster.
            if dfs(new_chosen, new_used, current_size + 1, remaining_pool[i+1:]):
                return True
                
        return False
        
    # We will run multiple restarts with different orderings to ensure we find the global max
    # If the first run finds 20, we stop.
    
    rng = random.Random(42)
    
    # Strategy 1: Ascending order
    # Strategy 2: Descending order
    # Strategy 3: Random orders
    
    # First, try a deterministic greedy-like ordering:
    # Sort by value (ascending)
    pool = pool_base.copy()
    dfs(chosen_mask, used_sums_mask, best_size, pool)
    
    if best_size >= 20:
        return sorted(best_solution)
        
    # If not found, try descending
    pool = pool_base.copy()
    pool.sort(reverse=True)
    dfs(chosen_mask, used_sums_mask, best_size, pool)
    
    if best_size >= 20:
        return sorted(best_solution)
        
    # Random restarts
    while time.time() < deadline:
        pool = pool_base.copy()
        rng.shuffle(pool)
        
        dfs(chosen_mask, used_sums_mask, best_size, pool)
        
        if best_size >= 20:
            break
            
    return sorted(best_solution)
