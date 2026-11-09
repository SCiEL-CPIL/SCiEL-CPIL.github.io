import numpy as np
import itertools
import random
import time

def construct():
    n = 192
    max_time = 55.0 # Leave 5s buffer
    
    # Initial solution: Greedy with random restarts to find a good start
    best_set = set()
    best_size = 0
    
    # Try a few random greedy initializations
    for _ in range(20):
        selected = []
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(x, selected):
                selected.append(x)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    # Also try descending greedy
    selected = []
    for x in range(n, 0, -1):
        if is_valid_add(x, selected):
            selected.append(x)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)

    start_time = time.time()
    
    while time.time() - start_time < max_time:
        current_set = list(best_set)
        
        # 1. Try to add any element (Single Add)
        added = False
        candidates = [x for x in range(1, n+1) if x not in best_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(x, current_set):
                best_set.add(x)
                best_size += 1
                added = True
                current_set.append(x)
                break
        
        if added:
            continue
            
        # 2. Try Swap 1-1
        swapped = False
        if best_set:
            # Pick a random element to remove
            y = random.choice(list(best_set))
            new_set = best_set - {y}
            current_set = list(new_set)
            
            candidates = [x for x in range(1, n+1) if x not in new_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(x, current_set):
                    best_set = new_set | {x}
                    # best_size unchanged
                    swapped = True
                    break
            
            if swapped:
                continue
                
        # 3. Try Swap 1-2 (Remove 1, Add 2)
        # This increases size by 1
        if best_set:
            y = random.choice(list(best_set))
            new_set = best_set - {y}
            current_set = list(new_set)
            
            candidates = [x for x in range(1, n+1) if x not in new_set and x != y]
            random.shuffle(candidates)
            
            # Filter candidates that are valid with new_set individually?
            # Not strictly necessary if we check pair validity, but speeds up.
            # Actually, if x1 is invalid with new_set, it can't be added.
            valid_x1 = []
            for x in candidates:
                if is_valid_add(x, current_set):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                # Try pairs from valid_x1
                # To avoid O(N^2) full checks, we can shuffle and check first few
                # Or just iterate
                for i in range(len(valid_x1)):
                    x1 = valid_x1[i]
                    # Check if x1 can be added to new_set (already known true)
                    # Now check if x2 can be added to new_set U {x1}
                    temp_set = current_set + [x1]
                    for j in range(i+1, len(valid_x1)):
                        x2 = valid_x1[j]
                        if is_valid_add(x2, temp_set):
                            best_set = new_set | {x1, x2}
                            best_size += 1
                            swapped = True
                            break
                    if swapped:
                        break
            
            if swapped:
                continue

        # 4. Try Swap 2-2 (Remove 2, Add 2) - Maintains size, helps escape local minima
        if len(best_set) >= 2:
            # Pick 2 random elements to remove
            y1, y2 = random.sample(list(best_set), 2)
            new_set = best_set - {y1, y2}
            current_set = list(new_set)
            
            candidates = [x for x in range(1, n+1) if x not in new_set and x != y1 and x != y2]
            random.shuffle(candidates)
            
            # Find 2 valid additions
            valid_adds = []
            for x in candidates:
                if is_valid_add(x, current_set):
                    valid_adds.append(x)
                    if len(valid_adds) == 2:
                        break
            
            if len(valid_adds) == 2:
                x1, x2 = valid_adds
                # Check interaction between x1 and x2
                if is_valid_add(x2, current_set + [x1]):
                    best_set = new_set | {x1, x2}
                    # best_size unchanged
                    swapped = True
            
            if swapped:
                continue

    return list(best_set)

def is_valid_add(x, selected_list):
    # selected_list is a list of ints
    # x is the new int
    
    if not selected_list:
        return True
        
    S = np.array(selected_list)
    
    # Check 1: x does not divide sum of any 3 in S
    if len(S) >= 3:
        # Generate all combinations of 3
        # For small S, itertools is okay, but let's try numpy if possible
        # Or just stick to itertools for correctness and simplicity, 
        # but the bottleneck was likely the Python loop overhead.
        # Let's use a vectorized check for triple sums.
        
        # We can compute all triple sums using broadcasting?
        # S[:, None, None] + S[None, :, None] + S[None, None, :]
        # This creates a 3D array of size S^3. 48^3 = 110k elements. Very fast in numpy.
        # But we need distinct indices i, j, k.
        # We can filter out i==j, j==k, i==k.
        
        # Alternatively, use itertools.combinations which is C-optimized.
        # Let's compare:
        # itertools: Python loop over ~18k combinations (48C3).
        # Numpy: Create 110k array, filter, check.
        
        # Let's try the Numpy approach for Check 1.
        S_3d = S[:, None, None] + S[None, :, None] + S[None, None, :]
        
        # Mask for distinct indices
        # i != j, j != k, i != k
        # Indices:
        i_idx, j_idx, k_idx = np.indices((len(S), len(S), len(S)))
        mask = (i_idx != j_idx) & (j_idx != k_idx) & (i_idx != k_idx)
        
        if np.any(mask):
            triple_sums = S_3d[mask]
            if np.any(triple_sums % x == 0):
                return False
    else:
        # If len < 3, no triples, so Check 1 passes
        pass
        
    # Check 2: For each a in S, a does not divide x + u + v
    # where u, v are distinct in S and u,v != a
    # This is equivalent to: For each a in S, let T = S \ {a}.
    # No pair (u,v) in T (u!=v) such that (x + u + v) % a == 0.
    
    for a in S:
        # T = S where S != a
        # Since S contains distinct integers, we can just mask
        T = S[S != a]
        if len(T) < 2:
            continue
            
        # We need to check if any pair sum from T is congruent to -x mod a
        # Pair sums: u + v for u, v in T, u != v
        # Use broadcasting on T
        T_2d = T[:, None] + T[None, :]
        
        # Mask for u != v (diagonal)
        i_idx, j_idx = np.indices((len(T), len(T)))
        mask = (i_idx != j_idx)
        
        if np.any(mask):
            pair_sums = T_2d[mask]
            # Check if (pair_sum + x) % a == 0
            if np.any((pair_sums + x) % a == 0):
                return False
                
    return True
