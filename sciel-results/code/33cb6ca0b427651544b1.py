"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill

    n = 256
    # Generate all 4-term arithmetic progressions
    # (a, a+d, a+2d, a+3d)
    aps = []
    ap_indices_by_pos = [[] for _ in range(n)]
    
    # 1-indexed logic for generation, 0-indexed for storage
    for d in range(1, 85):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            ap = (i0, i1, i2, i3)
            ap_idx = len(aps)
            aps.append(ap)
            
            ap_indices_by_pos[i0].append(ap_idx)
            ap_indices_by_pos[i1].append(ap_idx)
            ap_indices_by_pos[i2].append(ap_idx)
            ap_indices_by_pos[i3].append(ap_idx)
            
    total_aps = len(aps)
    
    # Precompute numpy arrays for fast delta computation
    aps_arr = np.array(aps, dtype=np.int32)  # shape (M, 4)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in ap_indices_by_pos]
    
    def count_mono(colors):
        """Count monochromatic 4-APs using numpy."""
        col = np.array(colors, dtype=np.int8)
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(k, colors):
        """
        Compute change in mono count if we flip colors[k].
        Uses numpy for speed on the affected APs.
        k is 0-indexed.
        """
        ap_indices = pos_aps_arr[k]
        if len(ap_indices) == 0:
            return 0
        
        # Get the affected APs
        affected = aps_arr[ap_indices]  # shape (K, 4)
        
        # Current colors at all positions in affected APs
        col = np.array(colors, dtype=np.int8)
        colors_affected = col[affected]  # shape (K, 4)
        
        # Check which are currently mono
        cur_mono = (colors_affected[:, 0] == colors_affected[:, 1]) & \
                   (colors_affected[:, 1] == colors_affected[:, 2]) & \
                   (colors_affected[:, 2] == colors_affected[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        
        # After flip: flip the specific position k in each AP
        # Find which column in each AP corresponds to k
        flipped = colors_affected.copy()
        for row_idx in range(len(ap_indices)):
            row = affected[row_idx]
            for col_idx in range(4):
                if row[col_idx] == k:
                    flipped[row_idx, col_idx] = 1 - flipped[row_idx, col_idx]
                    break
        
        new_mono = (flipped[:, 0] == flipped[:, 1]) & \
                   (flipped[:, 1] == flipped[:, 2]) & \
                   (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        
        return new_mono_count - cur_mono_count

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n (0-indexed)."""
        return [bin(i).count('1') % 2 for i in range(n)]

    # Use Thue-Morse as initial coloring
    best_colors = thue_morse(n)
    best_score = count_mono(best_colors)
    
    current_colors = best_colors[:]
    current_score = best_score
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.9998
    min_T = 0.005
    
    # Run SA loop with multiple restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current_colors = thue_morse(n)
            current_score = count_mono(current_colors)
        else:
            current_colors = best_colors[:]
            num_perturb = random.randint(10, 20)
            for _ in range(num_perturb):
                k = random.randint(0, n - 1)
                current_colors[k] = 1 - current_colors[k]
            current_score = count_mono(current_colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = current_colors[:]
        
        T = 1.0
        stuck_counter = 0
        no_improve_counter = 0
        
        while time.time() < deadline:
            # Pick random index
            k = random.randint(0, n - 1)
            delta = get_delta(k, current_colors)
            
            # Acceptance criterion
            accept = False
            if delta < 0:
                accept = True
                stuck_counter = 0
                no_improve_counter = 0
            else:
                if delta == 0:
                    accept = True
                    stuck_counter = 0
                else:
                    if T > 0.01:
                        prob = math.exp(-delta / T)
                    else:
                        prob = 0.0
                    if random.random() < prob:
                        accept = True
                        stuck_counter = 0
            
            if accept:
                current_colors[k] = 1 - current_colors[k]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                stuck_counter += 1
                no_improve_counter += 1
            
            # Cooling
            T *= cooling
            if T < min_T:
                T = 1.0
                stuck_counter = 0
                for _ in range(3):
                    k2 = random.randint(0, n - 1)
                    d2 = get_delta(k2, current_colors)
                    if d2 <= 0 or random.random() < 0.1:
                        current_colors[k2] = 1 - current_colors[k2]
                        current_score += d2
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = current_colors[:]
                            no_improve_counter = 0

            if stuck_counter > 3000 or no_improve_counter > 10000:
                break

    return best_colors
# EVOLVE-BLOCK-END
