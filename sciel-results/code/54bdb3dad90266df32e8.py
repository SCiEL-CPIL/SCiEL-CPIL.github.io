import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_idx] = tuple of other 3 indices
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
    
    # pos_in_ap[ap_idx][idx] = position of idx in ap
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors(colors):
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5, ...
    # We will use a deterministic greedy approach with local search refinement.
    for k in range(3, 15):
        if time.time() > deadline - 15:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Deterministic Greedy Construction with Lookahead
        # We try to assign colors 0..k-1 to positions 0..n-1.
        # To make it "greedy with lookahead", we don't just pick the first valid color,
        # but we can try to minimize future conflicts. However, a simple "First Fit" 
        # combined with strong local search is often sufficient and fast.
        
        # Let's use a few deterministic orderings (e.g., natural, reverse, shuffled by fixed seed)
        # to simulate "multiple restarts" deterministically.
        
        orderings = [
            list(range(n)),
            list(range(n-1, -1, -1)),
            [i for i in range(n) if i % 2 == 0] + [i for i in range(n) if i % 2 == 1], # Evens then Odds
            [i for i in range(n) if i % 3 == 0] + [i for i in range(n) if i % 3 == 1] + [i for i in range(n) if i % 3 == 2]
        ]
        
        for order in orderings:
            if time.time() > deadline - 10:
                break
                
            colors = np.full(n, -1, dtype=int)
            # Assign colors greedily
            # For each position in 'order', assign the smallest color in 0..k-1 
            # such that all APs containing this position that are now "complete" (all 4 assigned) 
            # or "almost complete" are satisfied? 
            # Actually, standard greedy: assign smallest color that doesn't immediately create a bad AP 
            # with already assigned neighbors.
            
            # To speed up, we check constraints only when all 4 positions of an AP are assigned.
            # But we need to ensure that when we assign a color, we don't make it *impossible* 
            # to fix later? That's hard. 
            # Instead, let's just do First-Fit Validity:
            # A color c is valid for position i if for every AP containing i, 
            # if the other 3 positions are already assigned, the resulting set of colors has size >= 3.
            # If other positions are not assigned, we assume they can be fixed later.
            
            valid = True
            for idx in order:
                assigned_color = -1
                for c in range(k):
                    # Check if color c is valid for idx
                    is_valid = True
                    for ap_idx in idx_to_aps[idx]:
                        pos = pos_in_ap[ap_idx][idx]
                        others = other_indices[ap_idx][pos]
                        # Check if all others are assigned
                        if all(colors[o] != -1 for o in others):
                            # Check constraint
                            c_others = [colors[o] for o in others]
                            full_colors = c_others + [c]
                            if len(set(full_colors)) <= 2:
                                is_valid = False
                                break
                    if is_valid:
                        assigned_color = c
                        break
                
                if assigned_color == -1:
                    valid = False
                    break
                colors[idx] = assigned_color
            
            if not valid:
                continue
            
            current_cost = count_bad_colors(colors)
            
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
            
            if best_k_cost == 0:
                break
                
        if best_k_cost == 0:
            result = list(best_k_solution)
            used_colors = set(result)
            
            # Try to reduce colors by merging
            # If we can merge color A into color B without creating bad APs, do it.
            improved = True
            while improved:
                if time.time() > deadline - 5:
                    break
                improved = False
                for color in list(used_colors):
                    for target_color in used_colors:
                        if color == target_color:
                            continue
                        # Try merging 'color' into 'target_color'
                        test_colors = np.array([target_color if c == color else c for c in result])
                        if count_bad_colors(test_colors) == 0:
                            result = list(test_colors)
                            used_colors = set(result)
                            improved = True
                            break
                    if improved:
                        break
                    if time.time() > deadline - 5:
                        break
            
            return result

    # If no valid solution found with k < 15, try a more aggressive local search from a random start
    # or just return a best effort.
    # Given the problem size, k=4 or k=5 should be sufficient.
    # Let's do a quick SA if we didn't find a solution yet.
    
    k = 5
    colors = np.random.randint(0, k, n)
    current_cost = count_bad_colors(colors)
    
    # Simple SA polish
    T = 1.0
    steps = 5000
    for _ in range(steps):
        if time.time() > deadline - 2:
            break
        i = np.random.randint(n)
        old_color = colors[i]
        new_color = np.random.randint(0, k)
        if new_color == old_color:
            continue
            
        delta = 0
        for ap_idx in idx_to_aps[i]:
            pos = pos_in_ap[ap_idx][i]
            others = other_indices[ap_idx][pos]
            c_others = [colors[j] for j in others]
            set_others = set(c_others)
            size_others = len(set_others)
            
            old_size = size_others + (0 if old_color in set_others else 1)
            new_size = size_others + (0 if new_color in set_others else 1)
            
            old_bad = 1 if old_size <= 2 else 0
            new_bad = 1 if new_size <= 2 else 0
            delta += new_bad - old_bad
            
        if delta < 0 or random.random() < np.exp(-delta / T):
            colors[i] = new_color
            current_cost += delta
            
        T *= 0.99
        if T < 0.01:
            T = 0.01
            
    if current_cost == 0:
        return list(colors)
        
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
