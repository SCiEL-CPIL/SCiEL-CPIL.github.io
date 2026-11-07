import random
import time

def _is_valid_sums(sums):
    """
    Check if the set of subset sums satisfies the no-divisibility constraint.
    sums: set of integers
    Returns True if no two distinct elements divide each other.
    """
    if len(sums) <= 1:
        return True
    
    # Optimization: Find max sum once
    max_sum = max(sums)
    
    # For each a in sums, check if any multiple of a exists in sums
    # We only need to check multiples 2a, 3a, ...
    for a in sums:
        # If a is 0, skip (but sums are positive)
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sums:
                return False
            multiple += a
    return True

def _greedy_construct(candidates, max_time=None):
    """
    Greedily construct a valid set from candidates.
    candidates: list of integers
    """
    current_set = []
    current_sums = set()
    
    for x in candidates:
        # Tentative addition
        # New sums are current_sums union (current_sums + x) union {x}
        
        # Optimization: If current_sums is empty, new_sums = {x}
        if not current_sums:
            new_sums = {x}
        else:
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
        
        # Check for collisions (duplicate sums)
        # If len(new_sums) != len(current_sums) + (len(current_sums) + 1), then collision?
        # Actually, if current_sums has size K, new sums should have size 2K + 1.
        # If size is less, there was a collision.
        expected_size = len(current_sums) * 2 + 1
        if len(new_sums) != expected_size:
            continue
            
        # Check divisibility
        if _is_valid_sums(new_sums):
            current_set.append(x)
            current_sums = new_sums
            
            if max_time and time.time() > max_time:
                break
                
    return current_set

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # Keep 5s margin for safety
    
    best_set = []
    
    # We know 1 is bad for size > 1. Start from 2.
    candidates_base = list(range(2, 257))
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Mix of random and structured orders
        # Random shuffle
        cands = candidates_base[:]
        random.shuffle(cands)
        
        # Occasionally try sorted orders
        if attempts % 10 == 0:
            cands = sorted(cands, reverse=True)
        elif attempts % 10 == 5:
            cands = sorted(cands)
            
        result = _greedy_construct(cands, max_time=deadline)
        
        if len(result) > len(best_set):
            best_set = result
            
    # Ensure types are correct
    # best_set is a list of ints.
    return list(best_set)
