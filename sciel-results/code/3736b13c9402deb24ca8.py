import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights
    weights = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # 1. Construct the [17, 8, 6] linear code
    # This is the shortened extended quadratic residue code.
    # We can find it by greedy linear construction.
    basis = []
    current_lin = [0]
    
    # Greedy linear construction
    for g in range(1, max_val):
        if len(basis) >= 8:
            break
        if time.time() > deadline - 10:
            break
            
        valid = True
        for c in current_lin:
            if weights[g ^ c] < MIN_D:
                valid = False
                break
                
        if valid:
            basis.append(g)
            old_code = current_lin
            current_lin = []
            for c in old_code:
                current_lin.append(c)
                current_lin.append(c ^ g)
                
    if len(current_lin) < 256:
        # Fallback: just return what we have
        return current_lin
        
    best_code = list(current_lin)
    best_size = len(best_code)
    
    # 2. Check for direct extension
    # Find x such that d(x, C) >= 6
    # We check all x not in C
    # Optimization: Use the fact that if d(x, C) < 6 for all x, we skip.
    
    # Let's try to find candidates for direct extension first.
    # This is fast if we use NumPy.
    # But 131k * 256 is large.
    # We'll skip direct extension if we suspect covering radius < 6.
    # Instead, go straight to "Remove 1, Add 2" strategy.
    
    # Strategy: For each c0 in C, remove it, find neighbors of c0 that are far from C\{c0}.
    # Add them greedily.
    
    code_set = set(best_code)
    code_arr = np.array(best_code, dtype=np.uint32)
    
    # Precompute neighbors for all c in C? No, do it on the fly.
    # Generate list of offsets with weight <= 5
    offsets = []
    for i in range(max_val):
        if weights[i] <= 5:
            offsets.append(i)
    offsets_arr = np.array(offsets, dtype=np.uint32)
    
    # Iterate over a subset of c0 to save time
    # We'll try all 256 if time permits.
    
    for c0 in best_code:
        if time.time() > deadline - 5:
            break
            
        # New code base: C \ {c0}
        base_code = [c for c in best_code if c != c0]
        base_set = set(base_code)
        
        # Find candidates x in neighborhood of c0 (dist <= 5)
        # such that d(x, base_code) >= 6
        
        # x = c0 ^ offset
        # We check x against base_code
        
        # Vectorized check:
        # x_vals shape (L,), base_code shape (M,)
        # We want min dist for each x.
        
        # Generate x_vals
        # Note: x must not be in base_code.
        # x = c0 ^ offset.
        # If x in base_code, d(x, base_code) = 0 < 6. So it will be filtered out.
        
        x_vals = c0 ^ offsets_arr
        
        # Compute distances
        # To save memory, do in chunks?
        # 9402 * 255 is small enough for a matrix?
        # 9402 * 255 * 4 bytes = ~9.5 MB. OK.
        
        xor_matrix = x_vals[:, np.newaxis] ^ np.array(base_code, dtype=np.uint32)[np.newaxis, :]
        dist_matrix = weights[xor_matrix]
        min_dists = np.min(dist_matrix, axis=1)
        
        # Filter
        valid_mask = min_dists >= MIN_D
        valid_x = x_vals[valid_mask]
        
        if len(valid_x) == 0:
            continue
            
        # Greedily add valid_x to base_code
        current_code = list(base_code)
        current_set = set(base_code)
        current_arr = np.array(base_code, dtype=np.uint32)
        
        # Sort valid_x by distance to c0? Or just iterate?
        # We want to maximize the number added.
        # Greedy: pick the one with max min-distance to current_code?
        # Or just pick any valid one.
        
        # Since they are all valid against base_code, we just need to check against each other.
        # We can sort them by weight or something?
        # Let's just iterate and add if valid against current_code.
        
        # To be efficient, we can check them all against each other?
        # No, greedy is fine.
        
        added_count = 0
        for x in valid_x:
            if x in current_set:
                continue
                
            # Check x against current_arr
            xor_vals = x ^ current_arr
            dists = weights[xor_vals]
            
            if np.min(dists) >= MIN_D:
                current_code.append(x)
                current_set.add(x)
                current_arr = np.append(current_arr, x)
                added_count += 1
                
        new_size = len(current_code)
        if new_size > best_size:
            best_code = current_code
            best_size = new_size
            
    return best_code
