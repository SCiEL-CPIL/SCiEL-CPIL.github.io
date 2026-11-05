import time
import random

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses backtracking with bitset-based sum collision detection,
    heuristic variable ordering, and multiple randomized restarts.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    # Precompute initial used sums from required elements
    used_sums_initial = 0
    for i in range(len(required)):
        for j in range(i, len(required)):
            s = required[i] + required[j]
            used_sums_initial |= (1 << s)
    
    # All candidate values not in required
    pool_base = [x for x in range(1, max_val + 1) if x not in required]
    
    best_solution = list(required)
    best_size = len(best_solution)
    
    def backtrack(current_set, used_sums, remaining_pool, current_size):
        nonlocal best_solution, best_size
        
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        if time.time() > deadline:
            return False
            
        if not remaining_pool:
            return False
            
        # Pruning: upper bound on achievable size
        if current_size + len(remaining_pool) <= best_size:
            return False
            
        # Build bitmask of current set for fast collision check
        current_set_mask = 0
        for y in current_set:
            current_set_mask |= (1 << y)
        
        # Try each candidate in the remaining pool
        for i, x in enumerate(remaining_pool):
            # Check if x creates any sum collision with existing elements
            # (x + y) for y in current_set must not be in used_sums
            if (used_sums & (current_set_mask << x)) != 0:
                continue
            # Check self-sum 2x
            if (used_sums & (1 << (2 * x))) != 0:
                continue
            
            # Valid: add x
            current_set.add(x)
            
            # Compute new sums mask
            new_sums_mask = (1 << (2 * x))
            for y in current_set:
                if y != x:
                    new_sums_mask |= (1 << (x + y))
            
            used_sums |= new_sums_mask
            
            # Recurse
            if backtrack(current_set, used_sums, remaining_pool[i+1:], current_size + 1):
                return True
            
            # Undo
            current_set.remove(x)
            used_sums &= ~new_sums_mask
            
        return False
    
    # Multiple restarts with different orderings
    rng = random.Random(12345)
    
    # First, try a deterministic greedy + backtrack approach
    # Order candidates by how "compatible" they are (fewer conflicts)
    
    # Try several different strategies
    strategies = [
        'ascending', 'descending', 'random', 'greedy_compat'
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        pool = pool_base.copy()
        
        if strategy == 'ascending':
            pass  # already sorted
        elif strategy == 'descending':
            pool.sort(reverse=True)
        elif strategy == 'random':
            rng.shuffle(pool)
        elif strategy == 'greedy_compat':
            # Sort by compatibility: prefer elements that don't conflict with required
            def compat_score(x):
                score = 0
                for r in required:
                    if (used_sums_initial & (1 << (x + r))) != 0:
                        score += 1
                    if (used_sums_initial & (1 << (2 * x))) != 0:
                        score += 10
                return score
            pool.sort(key=compat_score)
        
        # Run backtracking
        current_set = set(required)
        used_sums = used_sums_initial
        
        backtrack(current_set, used_sums, pool, len(required))
        
        if best_size >= 20:
            break
    
    # If we haven't found a large set, do more randomized restarts
    restart_count = 0
    while time.time() < deadline and best_size < 20:
        pool = pool_base.copy()
        rng.shuffle(pool)
        
        current_set = set(required)
        used_sums = used_sums_initial
        
        backtrack(current_set, used_sums, pool, len(required))
        restart_count += 1
        
        if best_size >= 20:
            break
    
    return sorted(best_solution)
