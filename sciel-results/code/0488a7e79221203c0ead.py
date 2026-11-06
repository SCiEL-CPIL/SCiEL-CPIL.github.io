"""Program using multi-start simulated annealing with diverse structured initializations and vectorized delta updates."""
import numpy as np
import random
import time

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    
    # Precompute APs
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap_array = np.array(ap_list, dtype=np.int32)
    num_aps = len(ap_array)
    
    # Mapping from position to list of AP indices
    pos_to_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        a, b, c, d_pos = ap_array[idx]
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d_pos].append(idx)
        
    # Convert to numpy arrays for faster access
    pos_to_aps_np = [np.array(lst, dtype=np.int32) for lst in pos_to_aps]

    def make_thue_morse():
        """Thue-Morse sequence: t(n) = popcount(n) % 2."""
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = bin(i).count('1') % 2
        return colors

    def make_block_coloring(block_size):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i // block_size) % 2
        return colors
    
    def make_mod_coloring(mod):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i % mod) % 2
        return colors

    def make_xor_pattern(x):
        """Color based on XOR of block index and sub-block index."""
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            block = i // x
            sub = (i % x) // (x // 2)
            colors[i] = block % 2 ^ sub
        return colors
        
    def make_random():
        return np.random.randint(0, 2, n, dtype=np.int8)

    # Generate diverse structured candidates
    candidates = []
    
    # Thue-Morse and inverted
    tm = make_thue_morse()
    candidates.append(tm)
    candidates.append(1 - tm)
    
    # Block sizes
    for bs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]:
        candidates.append(make_block_coloring(bs))
        
    # Mod-based
    for mod in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
        candidates.append(make_mod_coloring(mod))
        
    # XOR patterns
    for x in [4, 8, 16, 32, 64]:
        candidates.append(make_xor_pattern(x))
        
    # Random initializations
    for _ in range(10):
        candidates.append(make_random())
        
    # Find best initial coloring
    best_colors_init = None
    best_count_init = float('inf')
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count_init:
            best_count_init = c
            best_colors_init = cand.copy()
            
    # Multi-Start Simulated Annealing
    start_time = time.time()
    time_limit = 115  # seconds
    
    global_best_colors = best_colors_init.copy()
    global_best_count = best_count_init
    
    num_starts = 15
    time_per_start = (time_limit - 5) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Initialize current state
        if start_idx == 0:
            colors = global_best_colors.copy()
        else:
            # Perturb the global best by flipping a few random bits
            colors = global_best_colors.copy()
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                pos = random.randint(0, n - 1)
                colors[pos] = 1 - colors[pos]
        
        current_count = count_mono_4ap(colors, n)
        
        # SA Parameters
        T = 5.0
        T_min = 0.001
        # Increase max_iters significantly since delta calc is vectorized
        max_iters = 500000 
        cooling = (T_min / T) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit - 1:
                break
                
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            
            # Vectorized Delta Calculation
            ap_indices = pos_to_aps_np[pos]
            if len(ap_indices) > 0:
                # Get all 4 positions for these APs
                # shape: (num_aps_involving_pos, 4)
                ap_positions = ap_array[ap_indices]
                
                # Get colors of all 4 positions
                all_colors = colors[ap_positions] # shape (num_aps, 4)
                
                # Create a mask for 'pos'
                pos_mask = ap_positions == pos
                # ~pos_mask is True for the other 3 positions
                other_colors = all_colors[~pos_mask] # shape (3 * num_aps,)
                
                # Reshape to (num_aps, 3)
                other_colors = other_colors.reshape(-1, 3)
                
                # Count how many match old_val
                match_old = (other_colors == old_val).sum(axis=1)
                
                # Delta calculation
                # If match_old == 3: AP was mono(old), becomes non-mono. Delta = -1
                # If match_old == 0: AP was non-mono (all new), becomes mono(new). Delta = +1
                # Else: Delta = 0
                
                delta = np.where(match_old == 3, -1, 0) + np.where(match_old == 0, 1, 0)
                total_delta = np.sum(delta)
            else:
                total_delta = 0
            
            if total_delta <= 0:
                colors[pos] = 1 - old_val
                current_count += total_delta
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors.copy()
            else:
                if random.random() < np.exp(-total_delta / T):
                    colors[pos] = 1 - old_val
                    current_count += total_delta
            
            T *= cooling
            if T < T_min:
                T = T_min

    return global_best_colors.tolist()
