import time
import random

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses a CSP-style backtracking search with bitsets for pairwise sums,
    randomized variable ordering, and multiple restarts to escape local minima.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    # Candidates are all numbers in [1, 284] not in required
    all_candidates = list(range(1, max_val + 1))
    pool_base = [x for x in all_candidates if x not in required]
    
    # Initial state
    current_set = set(required)
    # Bitset representation of used sums
    # Sums range from 1+1=2 to 284+284=568
    used_sums_initial = 0
    for i in range(len(required)):
        for j in range(i, len(required)):
            s = required[i] + required[j]
            used_sums_initial |= (1 << s)
            
    best_solution = list(required)
    best_size = len(best_solution)
    
    def backtrack(current_set, used_sums, remaining_pool):
        nonlocal best_solution, best_size
        
        current_size = len(current_set)
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        if current_size == target_size:
            return True
            
        if time.time() > deadline:
            return False
            
        if not remaining_pool:
            return False
            
        # Pruning: if we can't reach target size
        if current_size + len(remaining_pool) < target_size:
            return False
            
        # Compute mask for current set for fast collision checks
        current_set_mask = 0
        for y in current_set:
            current_set_mask |= (1 << y)
            
        # Iterate through remaining pool
        for i, x in enumerate(remaining_pool):
            # Check validity using bitsets
            # Sum collision with existing elements:
            # We need to check if (x + y) is in used_sums for any y in current_set.
            # This is equivalent to (used_sums & (current_set_mask << x)) != 0
            if (used_sums & (current_set_mask << x)) != 0:
                continue
            
            # Check self-sum 2x
            if (used_sums & (1 << (2 * x))) != 0:
                continue
            
            # Valid candidate
            current_set.add(x)
            
            # Update used_sums
            new_sums_mask = 0
            for y in current_set:
                if y != x:
                    new_sums_mask |= (1 << (x + y))
            new_sums_mask |= (1 << (2 * x))
            
            used_sums |= new_sums_mask
            
            # Recurse
            if backtrack(current_set, used_sums, remaining_pool[i+1:]):
                return True
            
            # Undo
            current_set.remove(x)
            used_sums &= ~new_sums_mask
            
        return False

    # Run the search with multiple randomized restarts
    # The deterministic order often fails for this specific constraint set.
    # Randomizing the order of candidates helps explore different branches.
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Shuffle the pool
        pool = pool_base.copy()
        rng.shuffle(pool)
        
        # Reset state for this restart
        current_set = set(required)
        used_sums = used_sums_initial
        
        # Run backtracking
        # Note: passing a fresh copy of pool each time is fine.
        # The backtracking function modifies current_set and used_sums in place (with undo).
        backtrack(current_set, used_sums, pool)
        
        # If we found the target size, we can stop
        if best_size == target_size:
            break
            
    return sorted(best_solution)
