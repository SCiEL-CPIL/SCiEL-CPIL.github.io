import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    
    # Precompute 4-APs
    all_4aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    # Map element to list of 4-AP indices it belongs to
    elem_to_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_4aps[x].append(idx)
            
    # Precompute 3-APs
    all_3aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            all_3aps.append((a, a + d, a + 2 * d))
            
    # Map element to list of 3-AP indices it belongs to
    elem_to_3aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(idx)

    # State: use integer bitmask for fast operations
    # A_mask: bit i set means element i is in A (using 0-indexed bits for elements 1..N)
    A_mask = 0  # Python int as arbitrary-precision bitmask
    current_3ap_count = 0
    
    # Helper: check if element x is in A
    def in_A(x):
        return (A_mask >> (x-1)) & 1
    
    # Helper: add element x to A
    def add_to_A(x):
        nonlocal A_mask
        A_mask |= (1 << (x-1))
    
    # Helper: remove element x from A
    def remove_from_A(x):
        nonlocal A_mask
        A_mask &= ~(1 << (x-1))
    
    # Helper: Check if adding x keeps set 4-AP free
    # Assumes x is NOT in A
    def is_valid_add(x):
        for idx in elem_to_4aps[x]:
            ap = all_4aps[idx]
            cnt = 0
            for y in ap:
                if y != x and in_A(y):
                    cnt += 1
                    if cnt >= 3:
                        return False
        return True
        
    # Helper: Check if swapping x (out) and y (in) keeps set 4-AP free
    # Assumes x IS in A, y is NOT in A
    def is_valid_swap(x, y):
        for idx in elem_to_4aps[y]:
            ap = all_4aps[idx]
            cnt = 0
            for z in ap:
                if z != y:
                    if in_A(z) and z != x:
                        cnt += 1
                        if cnt >= 3:
                            return False
        return True

    # Helper: Calculate gain in 3-AP count if we add x to A
    # Assumes x NOT in A
    def gain_add(x):
        gain = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and in_A(y):
                    cnt += 1
            if cnt == 2:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-AP count if we remove x from A
    # Assumes x IS in A
    def loss_remove(x):
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for y in ap:
                if y != x and in_A(y):
                    cnt += 1
            if cnt == 2:
                loss += 1
        return loss

    # Helper: Calculate change in 3-AP count if we swap x (out) and y (in)
    # Assumes x IS in A, y is NOT in A
    def delta_swap(x, y):
        loss = 0
        for idx in elem_to_3aps[x]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != x and in_A(z):
                    cnt += 1
            if cnt == 2:
                loss += 1
                
        gain = 0
        for idx in elem_to_3aps[y]:
            ap = all_3aps[idx]
            cnt = 0
            for z in ap:
                if z != y and in_A(z) and z != x:
                    cnt += 1
            if cnt == 2:
                gain += 1
                
        return gain - loss

    # Greedy Initialization with multiple restarts
    best_set_mask = 0
    best_score = 0
    
    for restart in range(3):
        A_mask = 0
        current_3ap_count = 0
        remaining = set(range(1, N + 1))
        
        while remaining:
            best_x = None
            best_gain = -1
            for x in remaining:
                if is_valid_add(x):
                    g = gain_add(x)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            if best_x is None:
                break
            add_to_A(best_x)
            current_3ap_count += best_gain
            remaining.remove(best_x)
        
        if current_3ap_count > best_score:
            best_score = current_3ap_count
            best_set_mask = A_mask
    
    # Restore best state
    A_mask = best_set_mask
    current_3ap_count = best_score
    
    # Simulated Annealing with time-based cooling
    T_start = 10.0
    T_end = 0.05
    total_time = deadline - start_time
    
    step = 0
    while time.time() < deadline:
        step += 1
        # Time-based cooling
        elapsed = time.time() - start_time
        progress = min(elapsed / total_time, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        r = random.random()
        
        if r < 0.3:
            # Try Add
            x = random.randint(1, N)
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        add_to_A(x)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
        elif r < 0.6:
            # Try Remove
            x = random.randint(1, N)
            if in_A(x):
                delta = -loss_remove(x)
                if delta >= 0 or random.random() < np.exp(delta / T):
                    remove_from_A(x)
                    current_3ap_count += delta
                    if current_3ap_count > best_score:
                        best_score = current_3ap_count
                        best_set_mask = A_mask
        else:
            # Try Swap
            x = random.randint(1, N)
            y = random.randint(1, N)
            if in_A(x) and not in_A(y):
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta >= 0 or random.random() < np.exp(delta / T):
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        if current_3ap_count > best_score:
                            best_score = current_3ap_count
                            best_set_mask = A_mask
    
    # Final local search polish: try all single-element swaps and adds/removes
    A_mask = best_set_mask
    current_3ap_count = best_score
    improved = True
    while improved:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if not in_A(x):
                if is_valid_add(x):
                    delta = gain_add(x)
                    if delta > 0:
                        add_to_A(x)
                        current_3ap_count += delta
                        improved = True
        # Try removing elements
        for x in range(1, N + 1):
            if in_A(x):
                delta = -loss_remove(x)
                if delta > 0:
                    remove_from_A(x)
                    current_3ap_count += delta
                    improved = True
        # Try swaps
        active = [x for x in range(1, N + 1) if in_A(x)]
        inactive = [x for x in range(1, N + 1) if not in_A(x)]
        for x in active:
            for y in inactive:
                if is_valid_swap(x, y):
                    delta = delta_swap(x, y)
                    if delta > 0:
                        remove_from_A(x)
                        add_to_A(y)
                        current_3ap_count += delta
                        improved = True
                        break
            if improved:
                break
    
    if current_3ap_count > best_score:
        best_score = current_3ap_count
        best_set_mask = A_mask
    
    # Convert to list
    res = [i for i in range(1, N + 1) if (best_set_mask >> (i-1)) & 1]
    
    if not res:
        return [1]
        
    return res
