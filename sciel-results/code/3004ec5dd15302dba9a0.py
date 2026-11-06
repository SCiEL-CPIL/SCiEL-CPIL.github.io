import time
import random
import numpy as np

def _greedy_with_order(order, deadline):
    """Greedy selection following a given order of candidates."""
    selected = []
    sums = set()
    sel_arr = np.array([], dtype=np.int64)
    
    for x in order:
        if time.time() > deadline:
            break
        k = len(selected)
        
        # Generate all new sums involving x
        # 1. (x, x, x) -> 3x
        # 2. (x, x, y) for y in selected -> 2x + y
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        
        new_sums_list = [3 * x]
        
        if k > 0:
            # 2x + y for each y in selected
            new_sums_list.extend(2 * x + int(y) for y in sel_arr)
            
            # x + y + z for y <= z in selected
            for i in range(k):
                yi = int(sel_arr[i])
                for j in range(i, k):
                    zj = int(sel_arr[j])
                    new_sums_list.append(x + yi + zj)
        
        new_sums = set(new_sums_list)
        
        # Check for internal collisions (already handled by set, but check count)
        if len(new_sums) != len(new_sums_list):
            continue
        
        # Check for collisions with existing sums
        if new_sums & sums:
            continue
        
        selected.append(x)
        sel_arr = np.append(sel_arr, x)
        sums.update(new_sums)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = []
    best_size = 0
    
    # First try: natural order (deterministic baseline)
    natural_order = list(range(1, 4097))
    result = _greedy_with_order(natural_order, deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # If we still have time, try random restarts
    if time.time() < deadline - 5:
        rng = random.Random(42)
        
        while time.time() < deadline - 2:
            # Create a shuffled order
            order = list(range(1, 4097))
            rng.shuffle(order)
            
            result = _greedy_with_order(order, deadline)
            if len(result) > best_size:
                best = result
                best_size = len(result)
            
            # If we've found a large set, be more conservative with restarts
            if best_size >= 16:
                break
    
    return best
