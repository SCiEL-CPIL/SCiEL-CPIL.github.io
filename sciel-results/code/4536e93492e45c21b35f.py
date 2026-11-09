import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
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
    
    # Precompute for each index i, the list of AP indices where i appears
    # This allows fast delta computation when flipping bit i
    # ap_indices_by_pos[i] = list of (ap_idx, pos_in_ap)
    # But for delta, we just need to know which APs are affected.
    # An AP is monochromatic if all 4 bits are equal.
    # Flipping bit i changes the status of every AP containing i.
    
    # Let's precompute the list of AP indices for each position
    pos_to_aps = [[] for _ in range(n)]
    for ap_idx, ap in enumerate(aps_arr):
        for pos in ap:
            pos_to_aps[pos].append(ap_idx)
    
    # Convert to numpy arrays for faster access? Or just lists.
    # Lists are fine for small n.
    
    def count_mono(c):
        # c is array of 0/1
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
    
    # Initial coloring: Thue-Morse
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    current_score = count_mono(colors)
    best_score = current_score
    best_colors = colors.copy()
    
    # Greedy Coordinate Descent
    # In each sweep, iterate through all positions 0..n-1.
    # For each position, calculate the delta in score if we flip it.
    # If delta < 0, flip and update score.
    
    # To calculate delta quickly:
    # Let S be the current set of monochromatic APs.
    # Flipping bit at pos p affects APs in pos_to_aps[p].
    # For each affected AP, check if it was mono before and is mono after.
    # Delta = (new_mono_count) - (old_mono_count)
    # We can compute this by checking the status of each affected AP.
    
    max_sweeps = 10000
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
            
        improved = False
        
        # Randomize order to avoid bias? Or fixed order?
        # Fixed order is deterministic and often sufficient for GCD.
        # Let's try fixed order first.
        
        for p in range(n):
            # Calculate delta for flipping bit p
            # Affected APs:
            affected_aps = pos_to_aps[p]
            
            delta = 0
            for ap_idx in affected_aps:
                ap = aps_arr[ap_idx]
                
                # Get the 4 values
                v0 = colors[ap[0]]
                v1 = colors[ap[1]]
                v2 = colors[ap[2]]
                v3 = colors[ap[3]]
                
                # Check if mono before
                was_mono = (v0 == v1) and (v1 == v2) and (v2 == v3)
                
                # Simulate flip at p
                # Find which position in AP is p
                # ap[0] == p, ap[1] == p, etc.
                # We can just check each
                new_v0 = 1 - v0 if ap[0] == p else v0
                new_v1 = 1 - v1 if ap[1] == p else v1
                new_v2 = 1 - v2 if ap[2] == p else v2
                new_v3 = 1 - v3 if ap[3] == p else v3
                
                is_mono = (new_v0 == new_v1) and (new_v1 == new_v2) and (new_v2 == new_v3)
                
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            if delta < 0:
                colors[p] = 1 - colors[p]
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
        
        if not improved:
            break
            
    return [int(c) for c in best_colors]
