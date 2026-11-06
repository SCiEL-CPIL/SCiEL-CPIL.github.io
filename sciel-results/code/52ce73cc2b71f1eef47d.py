import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precompute APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_list = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # 2. Build Incidence Structures
    # pos_to_aps[i] = list of AP indices containing i
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for k in range(4):
            pos = ap_list[ap_idx, k]
            pos_to_aps[pos].append(ap_idx)
            
    # Convert to numpy arrays for faster access if needed, 
    # but lists of ints are fine for indexing.
    # For vectorized delta, we need to quickly get the "other" positions.
    # Let's precompute for each position i, an array of the other positions for each AP it's in.
    # pos_others[i] is a 2D array of shape (num_aps_in_i, 3)
    pos_others = []
    for i in range(N):
        if not pos_to_aps[i]:
            pos_others.append(np.array([], dtype=np.int32).reshape(0, 3))
        else:
            others = []
            for ap_idx in pos_to_aps[i]:
                ap = ap_list[ap_idx]
                # Find the index of i in ap
                # Since ap is small, linear search is fine
                idx = np.where(ap == i)[0][0]
                # Get the other 3
                other_positions = [ap[j] for j in range(4) if j != idx]
                others.append(other_positions)
            pos_others.append(np.array(others, dtype=np.int32))

    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        c = coloring[ap_list]  # shape (num_aps, 4)
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    def delta_1_flip(coloring, pos):
        """Compute delta in mono count if we flip coloring[pos]."""
        if not pos_to_aps[pos]:
            return 0
        cur_val = coloring[pos]
        new_val = 1 - cur_val
        
        others = pos_others[pos] # shape (m, 3)
        if len(others) == 0:
            return 0
            
        # Get values of others
        other_vals = coloring[others] # shape (m, 3)
        
        # Before: all 4 equal?
        # cur_val == other_vals[:,0] == other_vals[:,1] == other_vals[:,2]
        before_mono = (other_vals[:, 0] == cur_val) & (other_vals[:, 1] == cur_val) & (other_vals[:, 2] == cur_val)
        
        # After: all 4 equal?
        after_mono = (other_vals[:, 0] == new_val) & (other_vals[:, 1] == new_val) & (other_vals[:, 2] == new_val)
        
        delta = int(np.sum(after_mono) - np.sum(before_mono))
        return delta

    def delta_2_flip(coloring, pos1, pos2):
        """Compute delta in mono count if we flip coloring[pos1] and coloring[pos2]."""
        # Affected APs are those containing pos1 or pos2
        set1 = set(pos_to_aps[pos1])
        set2 = set(pos_to_aps[pos2])
        affected_aps = list(set1.union(set2))
        
        if not affected_aps:
            return 0
            
        ap_indices = np.array(affected_aps, dtype=np.int32)
        ap_data = ap_list[ap_indices] # shape (m, 4)
        
        # Get current values of all 4 positions for these APs
        vals = coloring[ap_data] # shape (m, 4)
        
        # Identify which positions in these APs are pos1 or pos2
        # We need to flip the value of pos1 and pos2 in the vals matrix
        # Create a mask for pos1 and pos2
        mask1 = (ap_data == pos1)
        mask2 = (ap_data == pos2)
        
        # Copy vals to modify
        new_vals = vals.copy()
        
        # Flip pos1
        new_vals[mask1] = 1 - new_vals[mask1]
        # Flip pos2
        new_vals[mask2] = 1 - new_vals[mask2]
        
        # Count mono before
        mono_before = (vals[:, 0] == vals[:, 1]) & (vals[:, 1] == vals[:, 2]) & (vals[:, 2] == vals[:, 3])
        
        # Count mono after
        mono_after = (new_vals[:, 0] == new_vals[:, 1]) & (new_vals[:, 1] == new_vals[:, 2]) & (new_vals[:, 2] == new_vals[:, 3])
        
        delta = int(np.sum(mono_after) - np.sum(mono_before))
        return delta

    def local_search_1_flip(coloring, max_rounds=100):
        """Local search using 1-bit flips."""
        coloring = coloring.copy()
        count = count_mono(coloring)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            best_delta = 0
            best_pos = -1
            
            # Find the best flip
            for pos in range(N):
                d = delta_1_flip(coloring, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            
            if best_pos != -1:
                coloring[best_pos] = 1 - coloring[best_pos]
                count += best_delta
                improved = True
            else:
                break
                
        return coloring, count

    def local_search_2_flip(coloring, max_rounds=50):
        """Local search using 2-bit flips."""
        coloring = coloring.copy()
        count = count_mono(coloring)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            improved = False
            best_delta = 0
            best_p1, best_p2 = -1, -1
            
            # Find the best 2-flip
            # To speed up, we can iterate through all pairs
            for p1 in range(N):
                for p2 in range(p1 + 1, N):
                    d = delta_2_flip(coloring, p1, p2)
                    if d < best_delta:
                        best_delta = d
                        best_p1 = p1
                        best_p2 = p2
                        
            if best_p1 != -1:
                coloring[best_p1] = 1 - coloring[best_p1]
                coloring[best_p2] = 1 - coloring[best_p2]
                count += best_delta
                improved = True
            else:
                break
                
        return coloring, count

    def full_local_search(coloring, max_rounds=10):
        """Alternate between 1-flip and 2-flip local search."""
        current = coloring.copy()
        count = count_mono(current)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            
            # 1-flip pass
            current, count = local_search_1_flip(current, max_rounds=20)
            
            # 2-flip pass
            current, count = local_search_2_flip(current, max_rounds=5)
            
        return current, count

    # Thue-Morse initialization
    def thue_morse(n):
        t = [0]
        for _ in range(n.bit_length()):
            t = t + [1 - x for x in t]
        return t[:n]
    
    coloring = np.array(thue_morse(N), dtype=np.int8)
    best_coloring = coloring.copy()
    best_count = count_mono(best_coloring)
    
    # Initial polish
    coloring, count = full_local_search(coloring, max_rounds=20)
    if count < best_count:
        best_count = count
        best_coloring = coloring.copy()
    
    # ILS Loop
    while time.time() < deadline - 10.0:
        # Perturb
        num_flips = np.random.randint(5, 15)
        flip_positions = np.random.choice(N, size=num_flips, replace=False)
        perturbed = best_coloring.copy()
        perturbed[flip_positions] = 1 - perturbed[flip_positions]
        
        # Local Search
        ls_coloring, ls_count = full_local_search(perturbed, max_rounds=10)
        
        if ls_count < best_count:
            best_count = ls_count
            best_coloring = ls_coloring.copy()
            
    # Final polish on best
    final_coloring, final_count = full_local_search(best_coloring, max_rounds=50)
    if final_count < best_count:
        best_count = final_count
        best_coloring = final_coloring
        
    return [int(c) for c in best_coloring]
