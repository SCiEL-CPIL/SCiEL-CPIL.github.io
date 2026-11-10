import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1
            aps.append((idx1, idx2, idx3, idx4))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)
    
    # Build index_to_aps: for each position, which APs contain it and at which slot
    index_to_aps = [[] for _ in range(N)]
    for ap_id, ap in enumerate(aps):
        for pos in range(4):
            index_to_aps[ap[pos]].append((ap_id, pos))
    
    # Vectorized count of monochromatic APs
    def count_mono_vec(colors):
        """colors: numpy array of 0/1 of length N. Returns count of mono 4-APs."""
        # colors[aps_arr] has shape (total_aps, 4)
        c = colors[aps_arr]
        # A 4-AP is monochromatic if all 4 values are equal
        # Check: c[:,0]==c[:,1] AND c[:,1]==c[:,2] AND c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
    
    # WalkSAT-style local search
    def walk_sat_search(colors, max_steps, rng):
        """
        WalkSAT-style search: repeatedly pick a violated (monochromatic) clause,
        flip a random variable in it, weighted by improvement.
        colors: list of 0/1
        Returns: (best_colors, best_score)
        """
        colors_np = np.array(colors, dtype=np.int8)
        current_score = count_mono_vec(colors_np)
        best_score = current_score
        best_colors = colors_np.copy()
        
        # Precompute for fast delta: for each position, list of (ap_id, pos)
        # We'll use this to compute delta when flipping position k
        
        for step in range(max_steps):
            # Find all violated (monochromatic) APs
            c = colors_np[aps_arr]
            mono_mask = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
            
            if not np.any(mono_mask):
                break  # No monochromatic APs, optimal!
            
            # Pick a random violated AP
            violated_indices = np.where(mono_mask)[0]
            ap_id = int(rng.integers(len(violated_indices)))
            ap_id = int(violated_indices[ap_id])
            ap = aps_arr[ap_id]
            
            # For each of the 4 positions in this AP, compute the delta if we flip it
            best_delta = float('inf')
            best_pos = 0
            deltas = []
            
            for pos in range(4):
                k = int(ap[pos])
                # Compute delta for flipping position k
                delta = 0
                for other_ap_id, other_pos in index_to_aps[k]:
                    other_ap = aps_arr[other_ap_id]
                    # Check if the other 3 positions are monochromatic
                    o1, o2, o3 = [], [], []
                    for j in range(4):
                        if j != other_pos:
                            o1.append(int(other_ap[j]))
                    o1, o2, o3 = o1[0], o1[1], o1[2]
                    
                    c_o1 = colors_np[o1]
                    c_o2 = colors_np[o2]
                    c_o3 = colors_np[o3]
                    
                    if c_o1 == c_o2 == c_o3:
                        if colors_np[k] == c_o1:
                            delta -= 1  # Currently mono, will become non-mono
                        else:
                            delta += 1  # Currently non-mono, will become mono
                
                deltas.append(delta)
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
            
            # WalkSAT: with probability p, pick the best (most improving) flip
            # Otherwise, pick randomly from the 4 positions
            p = 0.5  # probability of greedy choice
            if rng.random() < p:
                pos_to_flip = best_pos
            else:
                # Weighted random: prefer lower deltas
                weights = [max(1, -d + 2) for d in deltas]  # higher weight for more improving
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                pos_to_flip = 0
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        pos_to_flip = i
                        break
            
            # Apply the flip
            k = int(ap[pos_to_flip])
            colors_np[k] = 1 - colors_np[k]
            current_score += deltas[pos_to_flip]
            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors_np.copy()
        
        return best_colors.tolist(), best_score
    
    # Multi-start with diverse initializations
    rng = np.random.default_rng(42)
    
    best_overall_score = float('inf')
    best_overall_colors = None
    
    # Strategy 1: Several random restarts with WalkSAT
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Block pattern
            colors = [(i // 2) % 2 for i in range(N)]
        elif restart == 1:
            # Thue-Morse
            colors = [bin(i).count('1') % 2 for i in range(N)]
        elif restart == 2:
            # Periodic
            colors = [i % 2 for i in range(N)]
        elif restart == 3:
            # Random
            colors = [int(rng.integers(2)) for _ in range(N)]
        else:
            # Random
            colors = [int(rng.integers(2)) for _ in range(N)]
        
        # Run WalkSAT with a time budget
        step_budget = 5000
        best_colors, best_score = walk_sat_search(colors, step_budget, rng)
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_colors = best_colors
    
    # Strategy 2: Refine the best found with more aggressive search
    if best_overall_colors is not None and time.time() < deadline - 20:
        # Try to improve with longer WalkSAT runs
        for refine_iter in range(3):
            if time.time() > deadline - 10:
                break
            step_budget = 10000
            best_colors, best_score = walk_sat_search(best_overall_colors, step_budget, rng)
            if best_score < best_overall_score:
                best_overall_score = best_score
                best_overall_colors = best_colors
            else:
                break
    
    # Strategy 3: If we have time, try a hybrid: use SA on the best coloring
    if best_overall_colors is not None and time.time() < deadline - 15:
        colors_np = np.array(best_overall_colors, dtype=np.int8)
        current_score = count_mono_vec(colors_np)
        best_score = current_score
        best_colors = colors_np.copy()
        
        T = 0.5
        cooling_rate = 0.99995
        min_T = 0.01
        
        while time.time() < deadline - 5:
            if T < min_T:
                break
            
            k = int(rng.integers(N))
            c_k = colors_np[k]
            
            delta = 0
            for ap_id, pos in index_to_aps[k]:
                ap = aps_arr[ap_id]
                o1, o2, o3 = [], [], []
                for j in range(4):
                    if j != pos:
                        o1.append(int(ap[j]))
                o1, o2, o3 = o1[0], o1[1], o1[2]
                
                c_o1 = colors_np[o1]
                c_o2 = colors_np[o2]
                c_o3 = colors_np[o3]
                
                if c_o1 == c_o2 == c_o3:
                    if c_k == c_o1:
                        delta -= 1
                    else:
                        delta += 1
            
            if delta < 0:
                colors_np[k] = 1 - c_k
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors_np.copy()
            else:
                if T > 0.0 and delta < 100 * T:
                    if rng.random() < math.exp(-delta / T):
                        colors_np[k] = 1 - c_k
                        current_score += delta
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors_np.copy()
            
            T *= cooling_rate
        
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_colors = best_colors.tolist()
    
    return best_overall_colors
