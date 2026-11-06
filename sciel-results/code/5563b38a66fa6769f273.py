"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_sidon_and_cover(a, target_diffs):
    """Check if a is a Sidon set covering target_diffs."""
    n = len(a)
    if n > 32:
        return False
    if max(a) > 10**6:
        return False
    if 0 not in a:
        return False
    
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d <= 0:
                return False
            if d in diffs:
                return False
            diffs.add(d)
    return all(d in diffs for d in target_diffs)

def _search_with_bitmask(M, target_diffs, deadline):
    """
    DFS with bitmask diff tracking for speed.
    """
    initial_set = [0]
    initial_diffs_mask = 0
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
    remaining_mask = target_mask
    
    def dfs(current_set, used_diffs_mask, remaining_mask, depth):
        if remaining_mask == 0:
            return current_set[:]
        
        if len(current_set) >= 32:
            return None
            
        if time.time() > deadline:
            return None
        
        last = current_set[-1]
        
        # Generate candidates: x > last, x <= M
        # x must not create a duplicate difference
        potential_xs = set()
        for a in current_set:
            for d in target_diffs:
                if not (remaining_mask & (1 << d)):
                    continue
                x = a + d
                if x > last and x <= M:
                    potential_xs.add(x)
        
        if not potential_xs:
            return None

        candidates = []
        for x in potential_xs:
            conflict = False
            new_covered_mask = 0
            for a in current_set:
                d = x - a
                if d <= 0:
                    conflict = True
                    break
                if used_diffs_mask & (1 << d):
                    conflict = True
                    break
                if remaining_mask & (1 << d):
                    new_covered_mask |= (1 << d)
            
            if conflict:
                continue
            
            new_count = bin(new_covered_mask).count('1')
            # Prefer candidates that cover more remaining diffs, then smaller x
            candidates.append((x, -new_count))
        
        if not candidates:
            return None
            
        candidates.sort()
        
        for x, _ in candidates:
            new_diffs_mask = 0
            for a in current_set:
                d = x - a
                new_diffs_mask |= (1 << d)
            
            new_used = used_diffs_mask | new_diffs_mask
            new_remaining = remaining_mask & ~new_diffs_mask
            
            res = dfs(current_set + [x], new_used, new_remaining, depth + 1)
            if res is not None:
                return res
                
        return None

    return dfs(initial_set, initial_diffs_mask, remaining_mask, 0)

def _local_polish(solution, target_diffs, deadline):
    """
    Try to reduce the maximum element by swapping it out for a smaller valid element.
    Returns a potentially improved solution.
    """
    if not solution:
        return solution
    
    best = solution[:]
    best_max = max(best)
    
    # Try to replace the largest element with a smaller one
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        current = best[:]
        if not current:
            break
        
        largest_idx = len(current) - 1
        largest_val = current[largest_idx]
        
        # Try smaller values for the last element
        found = False
        for new_val in range(1, largest_val):
            # Must be greater than second-to-last element
            if len(current) > 1 and new_val <= current[-2]:
                continue
            
            # Check if replacing largest with new_val maintains Sidon + coverage
            candidate = current[:-1] + [new_val]
            if _check_sidon_and_cover(candidate, target_diffs):
                best = candidate
                best_max = new_val
                found = True
                break
        
        if not found:
            break
    
    return best

def _construct_fallback():
    """
    Construct a valid Sidon set covering 1..30 using a systematic greedy approach.
    This is a guaranteed valid fallback.
    """
    result = [0]
    used_diffs = set()
    remaining = set(range(1, 31))
    
    # Greedily add elements that cover the most remaining diffs
    next_x = 1
    while remaining and len(result) < 32 and next_x <= 10**6:
        # Try next_x
        valid = True
        new_diffs = set()
        for a in result:
            d = next_x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            # Check if this helps cover remaining diffs
            covers_any = any(d in remaining for d in new_diffs)
            if covers_any or not remaining:
                result.append(next_x)
                used_diffs.update(new_diffs)
                remaining -= new_diffs
                next_x += 1
            else:
                next_x += 1
        else:
            next_x += 1
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    best_solution = None
    
    # Try progressively larger upper bounds
    # Start with small M to find minimal diameter solutions
    for M in range(35, 200):
        if time.time() > deadline - 10:
            break
            
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 0.5:
            break
            
        # Allocate more time to smaller M values where solutions are more likely
        if M < 50:
            m_deadline = min(deadline - 10, time.time() + 8.0)
        elif M < 80:
            m_deadline = min(deadline - 10, time.time() + 5.0)
        else:
            m_deadline = min(deadline - 10, time.time() + 3.0)
        
        result = _search_with_bitmask(M, target_diffs, m_deadline)
        if result is not None:
            if _check_sidon_and_cover(result, target_diffs):
                best_solution = result
                break
    
    # If we found a solution, try to polish it
    if best_solution is not None:
        polish_deadline = min(deadline, time.time() + 15.0)
        best_solution = _local_polish(best_solution, target_diffs, polish_deadline)
    
    # If no solution found, return fallback
    if best_solution is None:
        best_solution = _construct_fallback()
    
    return best_solution
# EVOLVE-BLOCK-END
