"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices for delta computation
    # pos_aps[i] contains indices of APs where position i appears
    pos_aps = [[] for _ in range(n)]
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_aps[pos].append(idx)
    
    # Convert to numpy arrays for faster access
    pos_aps_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_aps]
    
    def count_mono(colours):
        """Count monochromatic 4-term APs. colours is a numpy array of 0/1."""
        # For each AP, check if all 4 positions have the same colour
        # colours[aps_arr] gives shape (M, 4)
        c = colours[aps_arr]
        # An AP is monochromatic if all 4 values are equal
        # Check: all equal to first element
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))
    
    def get_delta(pos, colours):
        """Compute the change in monochromatic count if we flip position pos."""
        # Get all APs containing this position
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) == 0:
            return 0
        
        # Get the colours of these APs
        c = colours[aps_arr[ap_indices]]
        
        # For each AP, count how many positions have the same colour as pos
        # Before flip: pos has colour colours[pos]
        # After flip: pos has colour 1 - colours[pos]
        
        # Count mono APs before flip
        old_mono = int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))
        
        # After flipping pos, we need to recompute
        # But we can compute the delta more efficiently
        # For each AP, determine if it was mono before and after flip
        
        # The position's index within each AP varies
        # Let's just recompute for simplicity with the small subset
        new_colours = colours.copy()
        new_colours[pos] = 1 - colours[pos]
        
        c_new = new_colours[aps_arr[ap_indices]]
        new_mono = int(np.sum((c_new[:, 0] == c_new[:, 1]) & (c_new[:, 1] == c_new[:, 2]) & (c_new[:, 2] == c_new[:, 3])))
        
        # But we only need to consider APs that contain pos
        # APs not containing pos are unaffected
        # So delta = new_mono - old_mono for the affected APs
        # But we need to be careful: old_mono and new_mono are only for affected APs
        # The total delta is new_mono - old_mono (since unaffected APs don't change)
        
        return new_mono - old_mono
    
    # Multi-start simulated annealing
    best_score = float('inf')
    best_colours = None
    
    # Initial solutions: random and some structured ones
    np.random.seed(42)
    
    # Start with a few different initializations
    inits = []
    
    # Random
    inits.append(np.random.randint(0, 2, n))
    
    # Thue-Morse like
    tm = np.array([(i & 1) ^ ((i >> 1) & 1) ^ ((i >> 2) & 1) ^ ((i >> 3) & 1) for i in range(n)], dtype=np.int32)
    inits.append(tm)
    
    # Block pattern
    block = np.zeros(n, dtype=np.int32)
    for i in range(n):
        block[i] = (i // 8) % 2
    inits.append(block)
    
    # Another random
    inits.append(np.random.randint(0, 2, n))
    
    # Another structured
    struct = np.zeros(n, dtype=np.int32)
    for i in range(n):
        struct[i] = (i % 4) // 2
    inits.append(struct)
    
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        colours = init.copy()
        current_score = count_mono(colours)
        
        if current_score < best_score:
            best_score = current_score
            best_colours = colours.copy()
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        steps = 0
        while T > T_min and time.time() < deadline - 5:
            # Pick a random position to flip
            pos = np.random.randint(0, n)
            delta = get_delta(pos, colours)
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                colours[pos] = 1 - colours[pos]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colours = colours.copy()
            
            T *= cooling
            steps += 1
    
    # Return the best solution
    # Convert numpy array to list of ints
    return [int(c) for c in best_colours]
# EVOLVE-BLOCK-END
