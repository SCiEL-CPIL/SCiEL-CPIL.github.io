import time
import sys
import random

def _validate_full(elements):
    """
    Full strict validation.
    """
    if not elements:
        return True
    
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    if len(set(elements)) != len(elements):
        return False
        
    sums = set([0])
    for x in elements:
        new_sums = set(s + x for s in sums)
        if sums & new_sums:
            return False
        sums |= new_sums
        
    non_empty_sums = sorted(s for s in sums if s > 0)
    if not non_empty_sums:
        return True
    
    max_sum = non_empty_sums[-1]
    
    in_set = [False] * (max_sum + 1)
    for s in non_empty_sums:
        in_set[s] = True
    
    for a in non_empty_sums:
        if a > max_sum // 2:
            break
        for m in range(2 * a, max_sum + 1, a):
            if in_set[m]:
                return False
    return True

def _can_add_fast(sums_set, candidate, sorted_sums):
    """
    Check if candidate can be added. Uses sorted_sums for early termination.
    """
    # Check for duplicate sums
    for s in sums_set:
        if s + candidate in sums_set:
            return False
    
    # Generate new sums
    new_sums = set(sums_set)
    new_sums.update(s + candidate for s in sums_set)
    
    # Check divisibility using the sorted list approach
    # The new sums are sorted_sums and [s + candidate for s in sorted_sums]
    # We need to check if any sum divides another
    
    # Build a set for O(1) lookup
    new_sums_set = new_sums
    
    # Get all non-zero sums
    non_zero = sorted(s for s in new_sums_set if s > 0)
    if not non_zero:
        return True
    
    max_s = non_zero[-1]
    
    # Use boolean array for fast lookup
    in_set = [False] * (max_s + 1)
    for s in non_zero:
        in_set[s] = True
    
    # Check divisibility
    for a in non_zero:
        if a > max_s // 2:
            break
        # Check if any multiple of a is in the set
        for m in range(2 * a, max_s + 1, a):
            if in_set[m]:
                return False
    
    return True

def construct():
    deadline = time.time() + 80.0
    
    best_elements = []
    
    def greedy_pass(order):
        current_elements = []
        current_sums = {0}
        sorted_sums = [0]
        
        for cand in order:
            if time.time() > deadline:
                break
            
            # Fast check
            if _can_add_fast(current_sums, cand, sorted_sums):
                new_sums = set(current_sums)
                new_sums.update(s + cand for s in current_sums)
                current_elements.append(cand)
                current_sums = new_sums
                sorted_sums = sorted(current_sums)
        
        return current_elements
    
    # Try descending order first
    order_desc = list(range(256, 0, -1))
    greedy_result = greedy_pass(order_desc)
    if len(greedy_result) > len(best_elements):
        best_elements = greedy_result
    
    # Try multiple random restarts with different strategies
    random.seed(42)
    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 10.0 and restarts < max_restarts:
        restarts += 1
        
        # Different strategies for each restart
        strategy = restarts % 4
        
        if strategy == 0:
            # Pure descending
            order = list(range(256, 0, -1))
        elif strategy == 1:
            # Random shuffle
            order = list(range(1, 257))
            random.shuffle(order)
        elif strategy == 2:
            # Descending with random perturbations
            order = list(range(256, 0, -1))
            for i in range(len(order) - 1):
                if random.random() < 0.1:
                    j = random.randint(i + 1, len(order) - 1)
                    order[i], order[j] = order[j], order[i]
        else:
            # Grouped: top 100 descending, then random
            top = list(range(256, 155, -1))
            rest = list(range(155, 0, -1))
            random.shuffle(rest)
            order = top + rest
        
        result = greedy_pass(order)
        if len(result) > len(best_elements):
            best_elements = result
    
    # Final validation
    if best_elements and not _validate_full(best_elements):
        best_elements = [20, 21, 23]
    
    return list(best_elements)

if __name__ == "__main__":
    print(construct())
