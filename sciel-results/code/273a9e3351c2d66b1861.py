import time


def _greedy_construct(order, deadline):
    """Greedy construction with a given element ordering."""
    selected = []
    MASK = (1 << 201) - 1
    reachable = 1
    
    for x in order:
        if time.time() > deadline:
            break
            
        new_reachable = reachable | (reachable << x)
        
        if not (new_reachable & (1 << 200)):
            selected.append(x)
            reachable = new_reachable & MASK
            
    return selected


def _try_add_back(selected, deadline):
    """Try to add back excluded elements to improve cardinality."""
    selected_set = set(selected)
    all_elements = set(range(1, 101))
    excluded = list(all_elements - selected_set)
    
    MASK = (1 << 201) - 1
    reachable = 1
    for x in selected:
        reachable = (reachable | (reachable << x)) & MASK
    
    improved = False
    for x in excluded:
        if time.time() > deadline:
            break
            
        new_reachable = reachable | (reachable << x)
        
        if not (new_reachable & (1 << 200)):
            selected.append(x)
            selected_set.add(x)
            reachable = new_reachable & MASK
            improved = True
    
    return selected, improved


def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try multiple greedy orderings
    best_selected = []
    
    # Ordering 1: descending (original)
    order1 = list(range(100, 0, -1))
    sel1 = _greedy_construct(order1, deadline)
    if len(sel1) > len(best_selected):
        best_selected = sel1
    
    if time.time() > deadline:
        return best_selected
    
    # Ordering 2: ascending
    order2 = list(range(1, 101))
    sel2 = _greedy_construct(order2, deadline)
    if len(sel2) > len(best_selected):
        best_selected = sel2
    
    if time.time() > deadline:
        return best_selected
    
    # Ordering 3: even numbers first, then odd
    order3 = list(range(2, 101, 2)) + list(range(1, 100, 2))
    sel3 = _greedy_construct(order3, deadline)
    if len(sel3) > len(best_selected):
        best_selected = sel3
    
    if time.time() > deadline:
        return best_selected
    
    # Ordering 4: large evens, large odds, small evens, small odds
    order4 = list(range(100, 0, -2)) + list(range(99, 0, -2))
    sel4 = _greedy_construct(order4, deadline)
    if len(sel4) > len(best_selected):
        best_selected = sel4
    
    if time.time() > deadline:
        return best_selected
    
    # Try to add back excluded elements
    final_selected, _ = _try_add_back(best_selected, deadline)
    
    return final_selected
