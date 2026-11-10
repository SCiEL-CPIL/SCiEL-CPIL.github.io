import time
import random

def _has_4ap(x, S):
    """
    Check if adding x to S creates a non-constant 4-AP.
    S is a set of integers.
    """
    # x is the new element.
    # A 4-AP is a, a+d, a+2d, a+3d.
    # x can be in any of the 4 positions.
    
    # 1. x is the first term: x, x+d, x+2d, x+3d
    # d must be such that x+3d <= 512
    max_d = (512 - x) // 3
    for d in range(1, max_d + 1):
        if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
            return True
            
    # 2. x is the second term: x-d, x, x+d, x+2d
    # x-d >= 1 => d <= x-1
    # x+2d <= 512 => d <= (512-x)//2
    max_d = min(x - 1, (512 - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
            return True
            
    # 3. x is the third term: x-2d, x-d, x, x+d
    # x-2d >= 1 => d <= (x-1)//2
    # x+d <= 512 => d <= 512-x
    max_d = min((x - 1) // 2, 512 - x)
    for d in range(1, max_d + 1):
        if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
            return True
            
    # 4. x is the fourth term: x-3d, x-2d, x-d, x
    # x-3d >= 1 => d <= (x-1)//3
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
            return True
            
    return False

def _greedy_expand(S, n, order=None):
    """
    Given a set S, try to add elements from 1..n in the given order (or natural)
    if they don't create a 4-AP.
    Returns the expanded set.
    """
    if order is None:
        order = range(1, n + 1)
    
    S_new = set(S)
    for x in order:
        if x not in S_new:
            if not _has_4ap(x, S_new):
                S_new.add(x)
    return S_new

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # 1. Initial Greedy Construction
    # Start with empty set and add in natural order
    S_best = set()
    S_best = _greedy_expand(S_best, n)
    
    # Try a few different initial orders to find a better start
    # e.g., reverse, odds first, evens first
    initial_orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]
    
    for order in initial_orders:
        S_candidate = _greedy_expand(set(), n, order)
        if len(S_candidate) > len(S_best):
            S_best = S_candidate
            
    # 2. Iterated Local Search
    # Randomly remove some elements, then greedily expand.
    # If the result is better, keep it.
    
    S_current = set(S_best)
    
    # Pre-compute a list of elements for random selection
    all_elements = list(range(1, n + 1))
    
    while time.time() < deadline:
        # Determine perturbation strength
        # Randomly choose how many elements to remove
        # Usually 1-5 is good
        num_remove = random.randint(1, 5)
        
        # Create a copy to perturb
        S_perturbed = set(S_current)
        
        # Remove random elements
        if len(S_perturbed) > num_remove:
            # Efficient removal: convert to list, shuffle, take first k, convert back to set?
            # Or just pop random items from a list copy
            S_list = list(S_perturbed)
            random.shuffle(S_list)
            to_remove = set(S_list[:num_remove])
            S_perturbed = S_perturbed - to_remove
        
        # Greedy expand
        # Use a random order for expansion to break symmetry
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_new = _greedy_expand(S_perturbed, n, order)
        
        if len(S_new) > len(S_best):
            S_best = S_new
            S_current = S_new
        else:
            # If not improved, revert to S_current (which is S_best if we didn't improve)
            # This is a "restart from best" strategy.
            # Alternatively, we could keep S_current and try to recover, 
            # but restart from best is simpler and often effective for this size.
            S_current = S_best
            
    # Final check: try a few more greedy expansions on S_best with different orders
    # to squeeze out any last elements
    for _ in range(10):
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_candidate = _greedy_expand(S_best, n, order)
        if len(S_candidate) > len(S_best):
            S_best = S_candidate
            
    return sorted(S_best)
