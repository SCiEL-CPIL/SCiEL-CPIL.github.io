import random
import time
import math

def construct():
    n = 512
    
    # Precompute all 4-APs
    # AP: (a, a+d, a+2d, a+3d)
    # d >= 1
    # a >= 1
    # a+3d <= n
    
    ap_list = []
    # Map element index (1-based) to list of AP indices it belongs to
    # We use 0-based index for elements in internal structures, but keep 1-based for logic clarity
    # element_to_aps[i] where i is 0..n-1 (representing i+1)
    element_to_aps = [[] for _ in range(n)]
    
    ap_idx = 0
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            # Store 0-based indices for array access
            ap_list.append((a - 1, b - 1, c - 1, dd - 1))
            
            # Add this AP index to the element lists
            for idx in (a - 1, b - 1, c - 1, dd - 1):
                element_to_aps[idx].append(ap_idx)
            
            ap_idx += 1
            
    num_aps = len(ap_list)
    
    # Helper to check validity and get size (for initialization)
    def greedy_init():
        S = set()
        # Use a fast check for greedy
        # We can reuse the logic from the original but optimized
        # Or just use the ap_counts structure? 
        # Greedy is sequential. 
        # Let's just do the standard greedy check as it's fast enough for 512
        def creates_4ap(S, x):
            # x is 1-based
            # Case 1: x, x+d, x+2d, x+3d
            max_d = (n - x) // 3
            for d in range(1, max_d + 1):
                if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                    return True
            # Case 2: x-d, x, x+d, x+2d
            max_d = min(x - 1, (n - x) // 2)
            for d in range(1, max_d + 1):
                if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                    return True
            # Case 3: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, n - x)
            for d in range(1, max_d + 1):
                if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                    return True
            # Case 4: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                    return True
            return False

        for i in range(1, n + 1):
            if not creates_4ap(S, i):
                S.add(i)
        return S

    initial_set = greedy_init()
    
    # State for SA
    in_set = [False] * n
    for x in initial_set:
        in_set[x - 1] = True
        
    ap_counts = [0] * num_aps
    for i in range(n):
        if in_set[i]:
            for ap_id in element_to_aps[i]:
                ap_counts[ap_id] += 1
                
    violations = sum(1 for c in ap_counts if c == 4)
    size = sum(in_set)
    
    best_set = set(initial_set)
    best_size = size
    best_violations = 0 # We start with valid set
    
    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    penalty = 100.0 # Energy = -size + penalty * violations
    
    start_time = time.time()
    time_limit = 55.0
    
    random.seed(42)
    
    # Precompute list of elements for random selection
    all_elements = list(range(n))
    
    def calculate_delta_add(x_idx):
        # x_idx is 0-based
        if in_set[x_idx]:
            return None
        delta_v = 0
        for ap_id in element_to_aps[x_idx]:
            if ap_counts[ap_id] == 3:
                delta_v += 1
        delta_size = 1
        return delta_size, delta_v

    def calculate_delta_remove(x_idx):
        # x_idx is 0-based
        if not in_set[x_idx]:
            return None
        delta_v = 0
        for ap_id in element_to_aps[x_idx]:
            if ap_counts[ap_id] == 4:
                delta_v -= 1
        delta_size = -1
        return delta_size, delta_v

    def apply_add(x_idx):
        nonlocal violations, size
        in_set[x_idx] = True
        for ap_id in element_to_aps[x_idx]:
            old_c = ap_counts[ap_id]
            new_c = old_c + 1
            ap_counts[ap_id] = new_c
            if old_c == 3 and new_c == 4:
                violations += 1
        size += 1

    def apply_remove(x_idx):
        nonlocal violations, size
        in_set[x_idx] = False
        for ap_id in element_to_aps[x_idx]:
            old_c = ap_counts[ap_id]
            new_c = old_c - 1
            ap_counts[ap_id] = new_c
            if old_c == 4 and new_c == 3:
                violations -= 1
        size -= 1

    while T > min_T:
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type: Add, Remove, or Swap
        # Bias towards adding if size is small, removing if large?
        # Let's just pick randomly.
        move_type = random.random()
        
        accepted = False
        if move_type < 0.4: # Add
            # Pick random element not in set
            # Optimization: just pick random index, check if in set
            # If in set, skip or pick another
            for _ in range(5): # Try a few times to find a valid add candidate
                x_idx = random.choice(all_elements)
                if not in_set[x_idx]:
                    break
            else:
                continue # All in set? Unlikely
                
            delta_size, delta_v = calculate_delta_add(x_idx)
            if delta_size is None:
                continue
                
            delta_E = -delta_size + penalty * delta_v
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                apply_add(x_idx)
                accepted = True
                if violations == 0 and size > best_size:
                    best_size = size
                    best_set = set(i + 1 for i in range(n) if in_set[i])
                    
        elif move_type < 0.7: # Remove
            # Pick random element in set
            for _ in range(5):
                x_idx = random.choice(all_elements)
                if in_set[x_idx]:
                    break
            else:
                continue # Set empty?
                
            delta_size, delta_v = calculate_delta_remove(x_idx)
            if delta_size is None:
                continue
                
            delta_E = -delta_size + penalty * delta_v
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                apply_remove(x_idx)
                accepted = True
                # Removing doesn't improve size, so no best update here unless we are stuck
                # But it might allow future adds
                
        else: # Swap
            # Pick x to remove, y to add
            for _ in range(5):
                x_idx = random.choice(all_elements)
                if in_set[x_idx]:
                    break
            else:
                continue
                
            for _ in range(5):
                y_idx = random.choice(all_elements)
                if not in_set[y_idx] and y_idx != x_idx:
                    break
            else:
                continue
                
            # Calculate combined delta
            # Remove x
            d_size_x, d_v_x = calculate_delta_remove(x_idx)
            # Add y
            d_size_y, d_v_y = calculate_delta_add(y_idx)
            
            if d_size_x is None or d_size_y is None:
                continue
                
            total_delta_size = d_size_x + d_size_y
            total_delta_v = d_v_x + d_v_y
            
            delta_E = -total_delta_size + penalty * total_delta_v
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Apply
                apply_remove(x_idx)
                apply_add(y_idx)
                accepted = True
                if violations == 0 and size > best_size:
                    best_size = size
                    best_set = set(i + 1 for i in range(n) if in_set[i])
            
        # Cool down
        T *= cooling
        
    # Final check: if best_set is valid and non-empty
    # It should be valid by construction of best_set update
    if not best_set:
        # Fallback to initial
        return sorted(initial_set)
        
    return sorted(best_set)
