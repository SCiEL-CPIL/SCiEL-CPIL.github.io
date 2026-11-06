"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(selected, existing_sums, x):
    """Check if x can be added to maintain B_3[1] property. Returns True if valid."""
    # Triples involving x:
    # 1. (x, x, x) -> 3x
    # 2. (y, x, x) for y in selected -> y + 2x
    # 3. (y, y, x) for y in selected -> 2y + x
    # 4. (y, z, x) for y <= z in selected -> y + z + x
    
    s = 3 * x
    if s in existing_sums:
        return False
    
    for y in selected:
        s1 = y + 2 * x
        if s1 in existing_sums:
            return False
        s2 = 2 * y + x
        if s2 in existing_sums:
            return False
    
    n = len(selected)
    for i in range(n):
        y = selected[i]
        for j in range(i, n):  # j >= i to handle (y,y,x) and (y,z,x)
            z = selected[j]
            s3 = y + z + x
            if s3 in existing_sums:
                return False
    
    return True

def _add_element(selected, existing_sums, x):
    """Add x to selected and update existing_sums."""
    selected.append(x)
    s = 3 * x
    existing_sums.add(s)
    for y in selected[:-1]:
        existing_sums.add(y + 2 * x)
        existing_sums.add(2 * y + x)
    n = len(selected)
    for i in range(n - 1):
        y = selected[i]
        for j in range(i, n - 1):
            z = selected[j]
            existing_sums.add(y + z + x)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    best_set = []
    
    # Strategy: Use geometric skeletons, then greedy insertion with smart ordering
    
    skeleton_strategies = [
        # Powers of 3
        lambda: [1, 3, 9, 27, 81, 243, 729, 2187],
        # Powers of 4
        lambda: [1, 4, 16, 64, 256, 1024, 4096],
        # Mixed: start small, then jump
        lambda: [1, 2, 6, 18, 54, 162, 486, 1458],
        # Start with 1, then factor ~3.2
        lambda: [1, 3, 10, 32, 102, 326, 1043, 3338],
        # Sparse arithmetic
        lambda: [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
        # Very sparse
        lambda: [1, 50, 100, 200, 400, 800, 1600, 3200],
    ]
    
    for seed in range(500):
        if time.time() > deadline - 2.0:
            break
        
        random.seed(seed)
        
        # Choose skeleton
        strat_idx = seed % len(skeleton_strategies)
        skeleton = skeleton_strategies[strat_idx]()
        
        # Sometimes perturb the skeleton
        if seed % 3 == 0:
            skeleton = [x + random.randint(-2, 2) for x in skeleton]
            skeleton = [x for x in skeleton if 1 <= x <= 4096]
            skeleton = list(set(skeleton))
        
        selected = []
        existing_sums = set()
        in_selected = set()
        
        # Add skeleton elements
        for x in sorted(skeleton):
            if x in in_selected or x > 4096:
                continue
            if _can_add(selected, existing_sums, x):
                _add_element(selected, existing_sums, x)
                in_selected.add(x)
        
        # Greedy insertion: try candidates in order of increasing "conflict likelihood"
        # Precompute which values are already used
        used = set(selected)
        
        # Try to add more elements, prioritizing values that are "far" from existing sums
        # We'll try in a few different orders
        
        # Order 1: ascending
        # Order 2: descending  
        # Order 3: random
        # Order 4: based on distance to nearest existing sum
        
        orders = []
        
        # Ascending
        orders.append(list(range(1, 4097)))
        
        # Descending
        orders.append(list(range(4096, 0, -1)))
        
        # Random
        rand_order = list(range(1, 4097))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
        # For each order, try to add elements
        for order_idx, order in enumerate(orders):
            if time.time() > deadline - 2.0:
                break
            
            for x in order:
                if time.time() > deadline - 2.0:
                    break
                
                if x in in_selected:
                    continue
                
                if _can_add(selected, existing_sums, x):
                    _add_element(selected, existing_sums, x)
                    in_selected.add(x)
            
            # After each order, if we improved, continue; else break
            if len(selected) > len(best_set):
                best_set = list(selected)
            else:
                break
        
        # Update best
        if len(selected) > len(best_set):
            best_set = list(selected)
        
        # Early stop if we've reached a very good size
        if len(best_set) >= 25:
            break
    
    return best_set

def _check_and_add(selected, existing_sums, x):
    """Check if x can be added to maintain B_3[1] property. If yes, add it and return True."""
    # Triples involving x:
    # 1. (x, x, x) -> 3x
    # 2. (y, x, x) for y in selected -> y + 2x
    # 3. (y, y, x) for y in selected -> 2y + x
    # 4. (y, z, x) for y < z in selected -> y + z + x
    
    new_sums = set()
    
    # 1. (x, x, x)
    s = 3 * x
    if s in existing_sums:
        return False
    new_sums.add(s)
    
    # 2 and 3: Iterate through selected
    for y in selected:
        # (y, x, x) -> y + 2x
        s1 = y + 2 * x
        if s1 in existing_sums or s1 in new_sums:
            return False
        new_sums.add(s1)
        
        # (y, y, x) -> 2y + x
        s2 = 2 * y + x
        if s2 in existing_sums or s2 in new_sums:
            return False
        new_sums.add(s2)
    
    # 4. (y, z, x) with y < z in selected
    n = len(selected)
    for i in range(n):
        y = selected[i]
        for j in range(i + 1, n):
            z = selected[j]
            s = y + z + x
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
    
    # All checks passed, add x
    selected.append(x)
    existing_sums.update(new_sums)
    return True
# EVOLVE-BLOCK-END
