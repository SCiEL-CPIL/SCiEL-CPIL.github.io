"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _reachable_sums(selected):
    """Compute set of reachable subset sums (capped at 200) for a set of integers."""
    r = {0}
    for x in selected:
        new = {s + x for s in r if s + x <= 200}
        r |= new
    return r

def _run_greedy(order):
    """
    Greedy selection: process elements in given order, add x if it doesn't create sum 200.
    Returns the selected list.
    """
    selected = []
    reachable = {0}
    for x in order:
        if 200 - x in reachable:
            continue
        selected.append(x)
        new_sums = {s + x for s in reachable if s + x <= 200}
        reachable |= new_sums
    return selected

def _local_search_add_only(start_set, deadline):
    """
    Local search: repeatedly try to add any excluded element that is safe.
    Returns improved set.
    """
    selected = set(start_set)
    excluded = set(range(1, 101)) - selected
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        reachable = _reachable_sums(selected)
        
        # Try adding any excluded element (largest first for potential benefit)
        for x in sorted(excluded, reverse=True):
            if 200 - x in reachable:
                continue
            # Check if adding x creates 200
            new_reach = set(reachable)
            for s in reachable:
                if s + x <= 200:
                    new_reach.add(s + x)
            if 200 not in new_reach:
                selected.add(x)
                excluded.discard(x)
                changed = True
                break
    
    return selected

def _local_search_swaps(start_set, deadline):
    """
    Local search: try swaps (remove one, add one) to maintain or improve size.
    Also try 2-for-1 and 1-for-2 swaps for cardinality gains.
    """
    selected = set(start_set)
    excluded = set(range(1, 101)) - selected
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps (maintain size, but might enable further additions)
        # Actually, 1-for-1 doesn't increase size directly, but might change the reachable set
        # to allow additions later. For now, focus on additions first via _local_search_add_only.
        # So this function focuses on structural changes that might enable growth.
        
        # Try 2-for-1: remove 2 elements, add 1 (net -1, bad for cardinality) - skip
        # Try 1-for-2: remove 1 element, add 2 (net +1, good for cardinality)
        
        # Iterate over elements to remove
        for to_remove in list(selected):
            if time.time() > deadline:
                break
            
            candidate = selected - {to_remove}
            reach = _reachable_sums(candidate)
            
            # Try adding pairs from excluded
            ex_list = sorted(excluded, reverse=True)
            found_pair = False
            for i in range(len(ex_list)):
                x1 = ex_list[i]
                if 200 - x1 in reach:
                    continue
                # Check if x1 can be added
                new_reach1 = set(reach)
                for s in reach:
                    if s + x1 <= 200:
                        new_reach1.add(s + x1)
                if 200 in new_reach1:
                    continue
                
                # Now try to add x2 from excluded (excluding x1)
                for j in range(len(ex_list)):
                    if i == j:
                        continue
                    x2 = ex_list[j]
                    if 200 - x2 in new_reach1:
                        continue
                    new_reach2 = set(new_reach1)
                    for s in new_reach1:
                        if s + x2 <= 200:
                            new_reach2.add(s + x2)
                    if 200 not in new_reach2:
                        # Success: remove to_remove, add x1 and x2
                        selected = candidate | {x1, x2}
                        excluded = set(range(1, 101)) - selected
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break
            if found_pair:
                break
        
        # After swaps, try to add more elements
        if improved and time.time() < deadline:
            selected = _local_search_add_only(selected, deadline)
            excluded = set(range(1, 101)) - selected
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    candidates = []
    
    # 1. Greedy Descending
    c1 = _run_greedy(range(100, 0, -1))
    candidates.append(c1)
    
    # 2. Greedy Ascending
    c2 = _run_greedy(range(1, 101))
    candidates.append(c2)
    
    # 3. Greedy with Skip Large (heuristic)
    # Skip >90 first, then add back
    order_skip = list(range(90, 0, -1)) + list(range(100, 90, -1))
    c3 = _run_greedy(order_skip)
    candidates.append(c3)
    
    # 4. Random Shuffles
    # Budget for random shuffles
    shuffle_deadline = deadline - 5.0
    max_shuffles = 5000
    base_order = list(range(1, 101))
    
    for _ in range(max_shuffles):
        if time.time() > shuffle_deadline:
            break
        random.shuffle(base_order)
        c_rand = _run_greedy(base_order)
        candidates.append(c_rand)
    
    # Find best candidate by cardinality
    best = max(candidates, key=len)
    best_size = len(best)
    
    # Local Search Phase 1: Add-only
    if time.time() < deadline - 2.0:
        improved = _local_search_add_only(best, deadline - 2.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Local Search Phase 2: Swaps (1-for-2)
    if time.time() < deadline - 1.0:
        improved = _local_search_swaps(best, deadline - 1.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Final verification
    reachable = _reachable_sums(best)
    if 200 in reachable:
        # Fallback to a known valid greedy result
        fallback = _run_greedy(range(100, 0, -1))
        return fallback
    
    return best
# EVOLVE-BLOCK-END
