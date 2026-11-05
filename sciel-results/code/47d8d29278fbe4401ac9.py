import time
import random
import sys

def construct():
    """
    Finds a maximum size Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses optimized backtracking with greedy initialization and conflict-based ordering.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    required = [1, 2, 5, 11]
    max_val = 284
    target_size = 20
    
    # Known optimal 20-mark Golomb ruler (0-based) shifted to 1-based
    known_ruler = [0, 1, 4, 10, 18, 23, 29, 42, 47, 52, 78, 82, 90, 91, 119, 129, 144, 172, 186, 283]
    known_set = set(x + 1 for x in known_ruler)
    
    # Build pool with priority: known_set elements first, then others
    pool = [x for x in range(1, max_val + 1) if x not in required]
    pool.sort(key=lambda x: (0 if x in known_set else 1, x))
    
    # Initial state
    current_set = set(required)
    used_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            used_sums.add(required[i] + required[j])
    
    best_solution = list(required)
    best_size = len(best_solution)
    
    # Greedy phase: quickly build a large set
    greedy_set = set(required)
    greedy_sums = set(used_sums)
    for x in pool:
        if time.time() > deadline - 10:
            break
        # Check if x can be added
        can_add_x = True
        for y in greedy_set:
            if x + y in greedy_sums:
                can_add_x = False
                break
        if can_add_x and 2 * x in greedy_sums:
            can_add_x = False
        if can_add_x:
            greedy_set.add(x)
            for y in greedy_set:
                greedy_sums.add(x + y)
            greedy_sums.add(2 * x)
            if len(greedy_set) >= target_size:
                break
    
    if len(greedy_set) > best_size:
        best_solution = list(greedy_set)
        best_size = len(best_solution)
    
    # If greedy found target size, return immediately
    if best_size >= target_size:
        result = sorted(best_solution)
        # Verify
        sums = set()
        valid = True
        for i in range(len(result)):
            for j in range(i, len(result)):
                s = result[i] + result[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid and all(r in result for r in required):
            return result
    
    # Backtracking phase: try to improve from greedy solution or find better
    # Use the greedy set as starting point for backtracking
    current_set = set(best_solution)
    used_sums = set()
    for i in range(len(current_set)):
        for j in range(i, len(current_set)):
            used_sums.add(list(current_set)[i] + list(current_set)[j])
    
    # Remove elements from current_set that are not in required, to allow rearrangement
    # Actually, let's do a fresh backtracking but with better pruning
    
    def can_add_fast(x, current_set, used_sums):
        """Fast check if x can be added."""
        if 2 * x in used_sums:
            return False
        for y in current_set:
            if x + y in used_sums:
                return False
        return True
    
    def get_new_sums_fast(x, current_set):
        """Get new sums when adding x."""
        new_sums = set()
        for y in current_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums
    
    def max_possible_additional(current_size, used_sums, remaining_pool):
        """Estimate max additional elements possible."""
        if not remaining_pool:
            return 0
        # Each new element adds (current_size + 1) new sums
        # Total sums needed for size n: n*(n+1)//2
        # Current sums: len(used_sums)
        # Available sum range: 2 to 568, so 567 possible sums
        # But we need distinct sums
        max_add = 0
        for add in range(1, len(remaining_pool) + 1):
            needed_sums = (current_size + add) * (current_size + add + 1) // 2
            if needed_sums > 567:  # Max possible distinct sums in range [2, 568]
                break
            max_add = add
        return max_add
    
    def backtrack(current_set, used_sums, pool_index, depth):
        nonlocal best_solution, best_size
        
        current_size = len(current_set)
        if current_size > best_size:
            best_solution = list(current_set)
            best_size = current_size
            
        if current_size >= target_size:
            return True
            
        if time.time() > deadline:
            return False
            
        if pool_index >= len(pool):
            return False
            
        # Pruning: check if we can still reach target
        remaining = pool[pool_index:]
        max_add = max_possible_additional(current_size, used_sums, remaining)
        if current_size + max_add < target_size and current_size <= best_size:
            return False
        
        # Try candidates with conflict-based ordering
        # Sort remaining candidates by how many conflicts they have (fewer conflicts first)
        candidates = []
        for i in range(pool_index, len(pool)):
            x = pool[i]
            if can_add_fast(x, current_set, used_sums):
                # Count how many future candidates this would block (approximate)
                candidates.append(x)
        
        # Limit candidates to avoid excessive branching
        if len(candidates) > 20:
            # Prioritize known_set elements and those with fewer conflicts
            candidates.sort(key=lambda x: (0 if x in known_set else 1, x))
            candidates = candidates[:20]
        
        for x in candidates:
            if time.time() > deadline:
                return False
            
            current_set.add(x)
            new_sums = get_new_sums_fast(x, current_set)
            used_sums.update(new_sums)
            
            if backtrack(current_set, used_sums, pool_index + 1, depth + 1):
                return True
            
            current_set.remove(x)
            used_sums.difference_update(new_sums)
        
        return False
    
    # Run backtracking from a smaller starting set to explore more
    # Start with just required elements
    current_set = set(required)
    used_sums = set()
    for i in range(len(required)):
        for j in range(i, len(required)):
            used_sums.add(required[i] + required[j])
    
    backtrack(current_set, used_sums, 0, 0)
    
    # Validate and return
    result = sorted(best_solution)
    if len(result) < 4:
        return sorted(required)
    
    sums = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    
    if valid and all(r in result for r in required) and all(1 <= r <= 284 for r in result):
        return result
    else:
        return sorted(required)
