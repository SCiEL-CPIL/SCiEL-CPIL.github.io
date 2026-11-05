import time
import random

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses a CSP-style backtracking search with bitsets for pairwise sums,
    dynamic variable ordering (heuristic based on spreading), and randomized restarts.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    # Candidates are all numbers in [1, 284] not in required
    all_candidates = list(range(1, max_val + 1))
    pool = [x for x in all_candidates if x not in required]
    
    # Initial state
    current_set = set(required)
    # Bitset representation of used sums
    # Sums range from 1+1=2 to 284+284=568
    used_sums = 0
    for i in range(len(required)):
        for j in range(i, len(required)):
            s = required[i] + required[j]
            used_sums |= (1 << s)
            
    best_solution = list(required)
    best_size = len(best_solution)
    
    # Pre-compute a mask for the required set for faster initialization if needed, 
    # but we start from scratch each time if we restart.
    
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
            
        # Pruning: if we can't reach target size even with all remaining
        if current_size + len(remaining_pool) < target_size:
            return False
            
        # Compute current_set_mask for fast conflict checking
        current_set_mask = 0
        for y in current_set:
            current_set_mask |= (1 << y)
            
        # Identify valid candidates
        valid_candidates = []
        for x in remaining_pool:
            # Check collision with existing sums
            # (used_sums & (current_set_mask << x)) checks if any x+y is in used_sums
            if (used_sums & (current_set_mask << x)) != 0:
                continue
            # Check 2x
            if (used_sums & (1 << (2 * x))) != 0:
                continue
            valid_candidates.append(x)
            
        # If we can't reach target size with valid candidates
        if current_size + len(valid_candidates) < target_size:
            return False
            
        if not valid_candidates:
            return False
            
        # Heuristic Ordering:
        # We want to spread elements out. 
        # Calculate distance to nearest element in current_set for each valid candidate.
        # Sort by descending distance (try far ones first to keep gaps large).
        # This is O(N*M) which is small (20*284).
        
        def get_min_dist(x):
            min_d = max_val
            for y in current_set:
                d = abs(x - y)
                if d < min_d:
                    min_d = d
            return min_d
            
        # To add some randomness and avoid deterministic traps, 
        # we can shuffle the valid candidates slightly or use a random tie-breaker.
        # Let's sort by distance, but add a small random noise to the distance to break ties randomly.
        random.seed(int(time.time() * 1000) % 10000) # Simple seed, or just use random.random
        
        scored_candidates = []
        for x in valid_candidates:
            d = get_min_dist(x)
            # Add random noise to encourage exploration
            score = d + random.random()
            scored_candidates.append((score, x))
            
        scored_candidates.sort(key=lambda item: -item[0])
        
        # Try candidates in this order
        # Optimization: If we have many candidates, we might want to limit the branching factor?
        # No, we must try all to be complete, but the order matters for finding a solution fast.
        
        for _, x in scored_candidates:
            # Validity is already checked, but double check just in case? No, it's deterministic.
            current_set.add(x)
            
            # Update used_sums
            new_sums_mask = 0
            for y in current_set:
                if y != x:
                    new_sums_mask |= (1 << (x + y))
            new_sums_mask |= (1 << (2 * x))
            
            used_sums |= new_sums_mask
            
            # Recurse
            # Filter remaining_pool to exclude x
            # Instead of slicing, we can pass the original list and skip x?
            # Slicing creates a new list. For depth 16, it's fine.
            # But we can be smarter: pass the set of remaining?
            # Let's stick to list slicing for simplicity, it's fast enough for N=280.
            remaining_pool_next = [y for y in remaining_pool if y != x]
            
            if backtrack(current_set, used_sums, remaining_pool_next):
                return True
            
            # Undo
            current_set.remove(x)
            used_sums &= ~new_sums_mask
            
        return False

    # Run the search with a few randomized restarts if the first one fails?
    # The heuristic above includes randomness, so a single run is effectively randomized.
    # However, if it gets stuck, we might want to restart.
    # Given the time limit, let's just run the randomized search once.
    # If it finds 20, great. If not, it returns the best found.
    
    found = backtrack(current_set, used_sums, pool)
    
    return sorted(best_solution)
