import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of (ap_idx, pos_in_ap)
    idx_to_aps_pos = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for pos, idx in enumerate(ap):
            idx_to_aps_pos[idx].append((i, pos))
    
    # Convert to numpy arrays for faster access
    idx_to_aps_pos_arr = []
    for i in range(n):
        if idx_to_aps_pos[i]:
            ap_idxs = np.array([x[0] for x in idx_to_aps_pos[i]], dtype=np.int32)
            positions = np.array([x[1] for x in idx_to_aps_pos[i]], dtype=np.int32)
            idx_to_aps_pos_arr.append((ap_idxs, positions))
        else:
            idx_to_aps_pos_arr.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))

    def init_matches(colors):
        """Compute matches for all APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        return matches

    def apply_move(colors, i, new_color, ap_matches, idx_to_aps_pos_arr, aps):
        """
        Apply a move: change colors[i] to new_color.
        Updates ap_matches in place.
        Returns delta in the number of bad APs.
        """
        delta_bad = 0
        ap_idxs, positions = idx_to_aps_pos_arr[i]
        m = len(ap_idxs)
        
        for idx_in_list in range(m):
            ap_idx = ap_idxs[idx_in_list]
            pos = positions[idx_in_list]
            
            old_matches = ap_matches[ap_idx]
            
            # Get AP indices
            ap = aps[ap_idx]
            
            # Get current colors
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Determine new colors
            if pos == 0:
                nc0, nc1, nc2, nc3 = new_color, c1, c2, c3
            elif pos == 1:
                nc0, nc1, nc2, nc3 = c0, new_color, c2, c3
            elif pos == 2:
                nc0, nc1, nc2, nc3 = c0, c1, new_color, c3
            else:
                nc0, nc1, nc2, nc3 = c0, c1, c2, new_color
                
            # Compute new matches
            new_matches = 0
            if nc0 == nc1: new_matches += 1
            if nc0 == nc2: new_matches += 1
            if nc0 == nc3: new_matches += 1
            if nc1 == nc2: new_matches += 1
            if nc1 == nc3: new_matches += 1
            if nc2 == nc3: new_matches += 1
            
            ap_matches[ap_idx] = new_matches
            
            was_bad = old_matches >= 2
            is_bad = new_matches >= 2
            
            if was_bad and not is_bad:
                delta_bad -= 1
            elif is_bad and not was_bad:
                delta_bad += 1
                
        return delta_bad

    best_solution = None
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
            
        # Simulated Annealing
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            ap_matches = init_matches(colors)
            current_bad = int(np.sum(ap_matches >= 2))
            
            if current_bad == 0:
                return [int(c) for c in colors]
            
            T = 2.0
            T_min = 0.01
            cooling = 0.999
            no_improve = 0
            max_no_improve = 5000
            max_iters = 50000
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                if current_bad == 0:
                    return [int(c) for c in colors]
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                
                if new_color == old_color:
                    continue
                    
                # Apply move
                delta = apply_move(colors, i, new_color, ap_matches, idx_to_aps_pos_arr, aps)
                colors[i] = new_color
                current_bad += delta
                
                if delta < 0:
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        pass # Accept
                    else:
                        # Reject: Revert
                        colors[i] = old_color
                        # Revert ap_matches
                        delta_revert = apply_move(colors, i, old_color, ap_matches, idx_to_aps_pos_arr, aps)
                        current_bad += delta_revert
                        no_improve += 1
                        
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if no_improve > max_no_improve:
                    break

    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
