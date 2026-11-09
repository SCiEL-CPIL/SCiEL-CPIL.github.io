import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 115.0  # 5s safety margin under 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps)
    
    # Build for each position x, the list of AP indices it participates in
    # and its role (0,1,2,3) in that AP
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, dd) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[dd].append((idx, 3))
    
    # Convert to numpy arrays for speed
    pos_ap_indices = []
    pos_ap_roles = []
    for x in range(n):
        if pos_aps[x]:
            arr = np.array(pos_aps[x], dtype=np.int32)
            pos_ap_indices.append(arr[:, 0])
            pos_ap_roles.append(arr[:, 1])
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_roles.append(np.array([], dtype=np.int32))
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        # colors: 1-D array of 0/1, length n
        # For each AP, check if all 4 positions have same color
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def delta_flip(colors, x):
        """Compute change in cost if we flip position x."""
        if len(pos_ap_indices[x]) == 0:
            return 0
        idxs = pos_ap_indices[x]
        roles = pos_ap_roles[x]
        # Current colors of the 4 positions in each AP
        c0 = colors[aps[idxs, 0]]
        c1 = colors[aps[idxs, 1]]
        c2 = colors[aps[idxs, 2]]
        c3 = colors[aps[idxs, 3]]
        
        # Before flip: is this AP monochromatic?
        before_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # After flip: position x changes color
        # Rebuild the 4-tuple with x flipped
        new_c0 = c0.copy()
        new_c1 = c1.copy()
        new_c2 = c2.copy()
        new_c3 = c3.copy()
        # Flip the role in the APs where x appears
        mask0 = (roles == 0)
        mask1 = (roles == 1)
        mask2 = (roles == 2)
        mask3 = (roles == 3)
        if mask0.any():
            new_c0[mask0] = 1 - new_c0[mask0]
        if mask1.any():
            new_c1[mask1] = 1 - new_c1[mask1]
        if mask2.any():
            new_c2[mask2] = 1 - new_c2[mask2]
        if mask3.any():
            new_c3[mask3] = 1 - new_c3[mask3]
        
        after_mono = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)
        
        return int(after_mono.sum() - before_mono.sum())
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def greedy_local_search(colors, max_iters=5000):
        """Greedy: repeatedly flip the position with the best (most negative) delta."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            # Compute delta for all positions
            best_delta = 0
            best_x = -1
            for x in range(n):
                d = delta_flip(colors, x)
                if d < best_delta:
                    best_delta = d
                    best_x = x
            if best_x == -1:
                break
            colors[best_x] = 1 - colors[best_x]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        return best_colors, best_cost
    
    def simulated_annealing(colors, start_temp=5.0, end_temp=0.1, max_iters=3000):
        """SA: accept worse moves with probability exp(-delta/temp)."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            x = random.randint(0, n - 1)
            d = delta_flip(colors, x)
            if d < 0 or random.random() < np.exp(-d / max(temp, 1e-10)):
                colors[x] = 1 - colors[x]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        
        return best_colors, best_cost
    
    # Multi-start: Thue-Morse + random perturbations
    best_overall = thue_morse(n)
    best_cost_overall = count_mono(best_overall)
    
    # Start 1: pure Thue-Morse
    c, cost = greedy_local_search(best_overall.copy(), max_iters=3000)
    if cost < best_cost_overall:
        best_cost_overall = cost
        best_overall = c
    c, cost = simulated_annealing(best_overall.copy(), max_iters=2000)
    if cost < best_cost_overall:
        best_cost_overall = cost
        best_overall = c
    
    # Additional starts with random perturbations of Thue-Morse
    tm = thue_morse(n)
    num_starts = 12
    for s in range(num_starts):
        if time.time() > deadline - 10.0:
            break
        # Perturb Thue-Morse with random flips
        colors = tm.copy()
        num_flips = random.randint(5, 30)
        flip_positions = random.sample(range(n), num_flips)
        for x in flip_positions:
            colors[x] = 1 - colors[x]
        
        c, cost = greedy_local_search(colors.copy(), max_iters=2000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
        if time.time() > deadline - 5.0:
            break
        c, cost = simulated_annealing(best_overall.copy(), max_iters=1000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Final greedy polish
    if time.time() < deadline - 3.0:
        c, cost = greedy_local_search(best_overall.copy(), max_iters=2000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    return [int(x) for x in best_overall]
