"""Improved B_3[1] construction: optimized greedy with incremental local search."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(x, selected, existing_sums):
    """Check if x can be added to selected without violating B_3[1] property."""
    n = len(selected)
    
    # (x, x, x)
    s = 3 * x
    if s in existing_sums:
        return False
    
    # (x, x, y) for y in selected
    for y in selected:
        s = 2 * x + y
        if s in existing_sums:
            return False
    
    # (x, y, y) for y in selected
    for y in selected:
        s = x + 2 * y
        if s in existing_sums:
            return False
    
    # (x, y, z) for y <= z in selected
    for i in range(n):
        y = selected[i]
        for j in range(i, n):
            z = selected[j]
            s = x + y + z
            if s in existing_sums:
                return False
    
    return True

def _add_element(x, selected, existing_sums):
    """Add x to selected and update existing_sums."""
    n = len(selected)
    
    # (x, x, x)
    existing_sums.add(3 * x)
    
    # (x, x, y) for y in selected
    for y in selected:
        existing_sums.add(2 * x + y)
    
    # (x, y, y) for y in selected
    for y in selected:
        existing_sums.add(x + 2 * y)
    
    # (x, y, z) for y <= z in selected
    for i in range(n):
        y = selected[i]
        for j in range(i, n):
            z = selected[j]
            existing_sums.add(x + y + z)
    
    selected.append(x)

def _remove_element(x, selected, existing_sums):
    """Remove x from selected and update existing_sums."""
    # Remove sums involving x
    n = len(selected)
    
    # (x, x, x)
    existing_sums.discard(3 * x)
    
    # (x, x, y) for y in selected
    for y in selected:
        if y != x: # x is in selected, so y can be x
            existing_sums.discard(2 * x + y)
        else:
            # This case is (x, x, x) which was handled above? 
            # No, _add_element adds (x,x,y) for ALL y in selected (before append).
            # So if y=x, it added 3x.
            # Wait, _add_element adds 3x explicitly.
            # And then loops y in selected (old selected).
            # So if x was not in selected before, y!=x always in the loop?
            # No, x is the new element.
            # In _remove_element, x IS in selected.
            # So we must be careful.
            
            pass # Handled by generic discard below?
            
    # It's safer to just discard all sums that could be formed with x.
    # But we need to know which sums to discard.
    # Sums involving x:
    # 3x
    # 2x + y (y in S, y != x? No, y in S \ {x} for the cross terms, but 3x is separate)
    # Actually, the set of sums is symmetric.
    # We can just iterate through all pairs/triples in the *remaining* set + x?
    # No, we need to remove sums that *contain* x.
    
    # Sums containing x:
    # 3x
    # 2x + y for y in S \ {x}
    # x + 2y for y in S \ {x}
    # x + y + z for y, z in S \ {x}, y <= z
    
    # Note: If y=x, 2x+y = 3x. x+2y = 3x.
    # If y=z=x, x+y+z = 3x.
    # So we just need to discard:
    existing_sums.discard(3 * x)
    
    remaining = [y for y in selected if y != x]
    
    for y in remaining:
        existing_sums.discard(2 * x + y)
        existing_sums.discard(x + 2 * y)
        
    for i in range(len(remaining)):
        y = remaining[i]
        for j in range(i, len(remaining)):
            z = remaining[j]
            existing_sums.discard(x + y + z)
            
    selected.remove(x)

def _greedy_build(order, deadline):
    """Build a B_3[1] set by greedily adding elements in the given order."""
    selected = []
    existing_sums = set()

    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        if _can_add(x, selected, existing_sums):
            _add_element(x, selected, existing_sums)

    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping elements using incremental updates."""
    selected = list(selected)
    existing_sums = set()
    
    # Rebuild existing_sums initially
    n = len(selected)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                existing_sums.add(selected[i] + selected[j] + selected[k])
    
    best_size = len(selected)
    best_set = list(selected)
    
    # Try removing one element and adding a better one
    # We iterate until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        for remove_idx in range(n):
            if time.time() > deadline:
                return best_set
            
            x_remove = selected[remove_idx]
            
            # Temporarily remove
            _remove_element(x_remove, selected, existing_sums)
            n = len(selected)
            
            # Try adding candidates
            # We want to add an element that wasn't there.
            # To increase size, we just need to add *any* valid element.
            # But if we remove one and add one, size stays same.
            # We want to increase size, so we should try to add *without* removing?
            # Greedy already tried that.
            # Local search usually swaps to find a better configuration that *allows* future additions.
            # OR, we can try to remove one and add *two*?
            # Let's stick to swaps first to escape local minima, then try to add.
            
            # Actually, if we remove x and add y, size is constant.
            # If we find a swap that allows *subsequent* additions, that's good.
            # But simpler: Try to remove x, then try to add *any* y.
            # If we can add y, size increases by 0 (removed 1, added 1).
            # We need to find a state where we can add *more*.
            
            # Strategy: Remove x. Try to add y.
            # If we add y, we have same size.
            # Then try to add z. If we can, size increases.
            
            # Let's try to find a swap that results in a "more open" set.
            # Heuristic: Prefer smaller elements? Larger elements?
            
            # For now, let's just try to swap x with any valid y.
            # If we find a swap, we accept it (hill climbing).
            # Then we re-run the "try to add" phase.
            
            found_swap = False
            # Try to add a random candidate or iterate through a subset
            # Iterating 1..4096 is slow.
            # Let's try a few random candidates.
            candidates = random.sample(range(1, 4097), min(100, 4096))
            for x_add in candidates:
                if x_add == x_remove:
                    continue
                if x_add in selected:
                    continue
                if _can_add(x_add, selected, existing_sums):
                    _add_element(x_add, selected, existing_sums)
                    found_swap = True
                    improved = True
                    break
            
            if not found_swap:
                # Revert removal
                _add_element(x_remove, selected, existing_sums)
                selected.sort()
                
    # After local search, try to add more
    # Greedy pass again
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        if x not in selected:
            if _can_add(x, selected, existing_sums):
                _add_element(x, selected, existing_sums)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = list(selected)

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    best = []
    best_size = 0
    
    # Strategy 1: Start with Power of 4 skeleton
    skeleton = [4**i for i in range(7) if 4**i <= 4096]
    # Skeleton: 1, 4, 16, 64, 256, 1024, 4096
    # This is a valid B_3[1] set.
    selected = []
    existing_sums = set()
    for x in skeleton:
        _add_element(x, selected, existing_sums)
    
    # Fill gaps greedily
    # Try adding elements in increasing order
    for x in range(1, 4097):
        if time.time() > deadline - 10.0:
            break
        if x in selected:
            continue
        if _can_add(x, selected, existing_sums):
            _add_element(x, selected, existing_sums)
            
    if len(selected) > best_size:
        best = selected
        best_size = len(selected)
        
    # Local search on the result
    improved = _local_search(best, deadline)
    if len(improved) > best_size:
        best = improved
        best_size = len(best)
        
    # Random restarts with different skeletons or orders
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
            
        # Randomize the fill order
        candidates = [x for x in range(1, 4097) if x not in skeleton]
        random.shuffle(candidates)
        
        selected = []
        existing_sums = set()
        for x in skeleton:
            _add_element(x, selected, existing_sums)
            
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            if _can_add(x, selected, existing_sums):
                _add_element(x, selected, existing_sums)
                
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
            
        # Local search
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    return best
# EVOLVE-BLOCK-END
