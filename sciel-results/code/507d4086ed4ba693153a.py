"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_all_sums(selected):
    """
    Compute the set of all unordered triple sums for a given set of integers.
    Returns None if a collision is found, otherwise returns the set of sums.
    """
    n = len(selected)
    if n == 0:
        return set()
    
    sums = set()
    # i <= j <= k
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = selected[i] + selected[j] + selected[k]
                if s in sums:
                    return None
                sums.add(s)
    return sums

def _check_validity(selected):
    """Check if the set has unique triple sums."""
    return _compute_all_sums(selected) is not None

def _can_add(selected, x, used_sums):
    """
    Check if x can be added to selected without causing a sum collision.
    selected is a sorted list.
    used_sums is a set of all triple sums from selected.
    """
    n = len(selected)
    new_sums = set()
    
    # 1. (x, x, x)
    s = 3 * x
    if s in used_sums:
        return False
    new_sums.add(s)
    
    # 2. (a, x, x) for a in selected
    for a in selected:
        s = a + 2 * x
        if s in used_sums or s in new_sums:
            return False
        new_sums.add(s)
        
    # 3. (a, b, x) for a <= b in selected
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            s = a + b + x
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
            
    return True

def _add_sums(selected, x, used_sums):
    """Add all new sums involving x to used_sums."""
    n = len(selected)
    used_sums.add(3 * x)
    for a in selected:
        used_sums.add(a + 2 * x)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            b = selected[j]
            used_sums.add(a + b + x)

def _local_improve(selected, used_sums, limit, time_budget, max_size=80):
    """
    Try to improve the current set by swapping elements or adding new ones.
    This is a local search step.
    """
    start_time = time.time()
    best_selected = list(selected)
    best_sums = set(used_sums)
    
    # Strategy: Try to remove a large element and add two smaller ones? 
    # Or just try to add more elements if possible.
    # Since we are maximizing cardinality, and we are likely stuck at a local optimum
    # where no single addition is possible, we might need to swap.
    
    # Simple local search: try to swap an existing element x with a new element y
    # such that the set remains valid and we might be able to add more later.
    # Actually, just trying to add is the primary goal. If we can't add, we are stuck.
    # A more powerful move: remove one element, try to add two.
    
    # Given time constraints, a simple "try to add" loop with randomized candidate order
    # is already done in the main loop. 
    # Let's try a specific heuristic: 
    # If we are stuck, try removing the largest element and see if we can add two smaller ones.
    
    if len(best_selected) >= max_size:
        return best_selected, best_sums

    # Try to find a swap that allows adding another element
    # This is complex. Let's stick to a simpler heuristic:
    # Randomly shuffle the order of candidates for addition in the main loop is better.
    
    # For now, return the same set, as the main loop handles the addition.
    # This function is a placeholder for future complex local search.
    return best_selected, best_sums

def _construct_greedy_with_bias(limit, max_size, bias, time_limit, seed):
    """
    Greedy construction with a specific bias on candidate selection.
    bias: 'small' prefers smaller numbers, 'random' is random, 'large' prefers larger.
    """
    random.seed(seed)
    candidates = list(range(1, limit + 1))
    
    if bias == 'small':
        # Sort by value, but with some noise
        candidates.sort(key=lambda x: x + random.uniform(0, 50))
    elif bias == 'large':
        # Sort by -value, but with some noise
        candidates.sort(key=lambda x: -x + random.uniform(0, 50))
    else: # random
        random.shuffle(candidates)
        
    selected = []
    used_sums = set()
    
    for x in candidates:
        if len(selected) >= max_size:
            break
        if time.time() > time_limit:
            break
        if _can_add(selected, x, used_sums):
            selected.append(x)
            _add_sums(selected, x, used_sums)
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Run multiple random greedy attempts with different biases
    # We try 'small', 'random', and 'large' biases with different seeds.
    # 'small' is often best for maximizing count in additive bases.
    
    strategies = [
        ('small', 1), ('small', 2), ('small', 3),
        ('random', 1), ('random', 2),
        ('large', 1),
    ]
    
    for bias, seed in strategies:
        if time.time() > deadline - 5:
            break
            
        current_best_size = len(best_result)
        
        # Allocate time based on remaining budget
        # If we have plenty of time, give each strategy a fair chunk.
        # Otherwise, just run until deadline.
        strategy_deadline = deadline - 2.0
        
        selected = _construct_greedy_with_bias(limit, max_size, bias, strategy_deadline, seed)
        
        if len(selected) > current_best_size:
            best_result = selected
            
        # If we reached max size, we can stop early
        if len(best_result) >= max_size:
            break
            
    # Final check: if we have time, try to improve the best result with a different ordering
    if len(best_result) < max_size and time.time() < deadline - 10:
        # Try a pure ascending order greedy, which is the standard baseline
        # but with a fresh start.
        # Actually, the 'small' bias with noise is similar.
        # Let's try a specific deterministic greedy: 1, 2, 3, ...
        selected = []
        used_sums = set()
        for x in range(1, limit + 1):
            if len(selected) >= max_size:
                break
            if time.time() > deadline - 2:
                break
            if _can_add(selected, x, used_sums):
                selected.append(x)
                _add_sums(selected, x, used_sums)
        
        if len(selected) > len(best_result):
            best_result = selected

    return best_result

# EVOLVE-BLOCK-END
