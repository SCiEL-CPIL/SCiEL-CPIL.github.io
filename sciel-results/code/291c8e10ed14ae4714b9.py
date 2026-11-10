import time
import random

def _is_valid(selected, target=200):
    """Check if any subset of selected sums to target."""
    # Use a set of reachable sums
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= target}
        if target in new:
            return False
        reachable |= new
    return True

def _greedy_construct(order, target=200, limit=100):
    """Greedy construction given a specific order of elements."""
    selected = []
    reachable = {0}
    for x in order:
        if x > limit:
            continue
        new = {s + x for s in reachable if s + x <= target}
        if target not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_search_improve(initial_set, target=200, limit=100, time_limit=5.0):
    """
    Improve the set by removing an element and then greedily adding others.
    """
    best_set = list(initial_set)
    best_size = len(best_set)
    start_time = time.time()
    
    # Elements not in the set
    all_elements = set(range(1, limit + 1))
    
    # Try removing each element from the current best set
    # To be efficient, we only do this if we have time
    for a in list(best_set):
        if time.time() - start_time > time_limit:
            break
        
        # Remove a
        temp_set = [x for x in best_set if x != a]
        
        # Try to refill
        # Sort remaining candidates (all elements not in temp_set) in descending order
        candidates = [x for x in range(limit, 0, -1) if x not in temp_set]
        
        new_set = _greedy_construct(candidates, target, limit)
        # Note: _greedy_construct starts with empty reachable, so it will rebuild from scratch.
        # This is expensive (100 * 200 ops). 100 removals * 100 candidates = 10k ops * 100 = 1M ops.
        # This is fast enough.
        
        if len(new_set) > best_size:
            best_set = new_set
            best_size = len(new_set)
            
    return best_set

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 50.0  # 50s out of 60s
    
    # 1. Standard Greedy (Top-down)
    order_top = list(range(100, 0, -1))
    set_top = _greedy_construct(order_top)
    best_set = set_top
    best_size = len(best_set)
    
    # 2. Standard Greedy (Bottom-up)
    order_bot = list(range(1, 101))
    set_bot = _greedy_construct(order_bot)
    if len(set_bot) > best_size:
        best_set = set_bot
        best_size = len(set_bot)
        
    # 3. Randomized Greedy
    random.seed(42)
    num_tries = 20
    for i in range(num_tries):
        if time.time() - start_time > deadline:
            break
        random_order = list(range(1, 101))
        random.shuffle(random_order)
        set_rand = _greedy_construct(random_order)
        if len(set_rand) > best_size:
            best_set = set_rand
            best_size = len(set_rand)
            
    # 4. Local Search on the best set found
    if time.time() - start_time < deadline - 5:
        improved_set = _local_search_improve(best_set, time_limit=deadline - time.time() - 2)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Final check
    if not best_set:
        # Fallback: just return [1]
        return [1]
        
    return best_set
