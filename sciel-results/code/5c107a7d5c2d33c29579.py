"""Combined approach: deterministic Hamming-weight construction + simulated annealing refinement."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    N = 128
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            if i3 < N:
                ap_id = len(aps)
                aps.append((i0, i1, i2, i3))
                ap_indices[i0].append(ap_id)
                ap_indices[i1].append(ap_id)
                ap_indices[i2].append(ap_id)
                ap_indices[i3].append(ap_id)
    
    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations
    
    def hamming_weight(x):
        """Count the number of 1s in binary representation."""
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count
    
    # Strategy 1: Deterministic Hamming-weight construction
    # Map position i to its Hamming weight (0-7 for 0-127)
    hw_coloring = [hamming_weight(i) for i in range(N)]
    hw_violations = count_violations(hw_coloring)
    
    if hw_violations == 0:
        return hw_coloring
    
    # Strategy 2: Simulated annealing starting from various initializations
    best_coloring = None
    best_num_colors = N + 1
    
    # Try different numbers of colors, starting from small
    for k in range(3, 20):
        if time.time() > deadline - 5.0:
            break
            
        # Multiple restarts for this k
        for restart in range(3):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize
            if restart == 0:
                # Start from Hamming-weight coloring, mapped to k colors
                coloring = [hw_coloring[i] % k for i in range(N)]
            elif restart == 1:
                # Start from a pattern based on modular arithmetic
                coloring = [(i * 7) % k for i in range(N)]
            else:
                # Random initialization
                random.seed(restart * 1000 + k)
                coloring = [random.randint(0, k - 1) for _ in range(N)]
            
            current_violations = count_violations(coloring)
            best_local_coloring = coloring[:]
            best_local_violations = current_violations
            
            # Simulated Annealing
            T = 3.0
            cooling = 0.9995
            max_steps = 30000
            steps = 0
            
            while current_violations > 0 and steps < max_steps and time.time() < deadline - 1.0:
                steps += 1
                idx = random.randint(0, N - 1)
                old_color = coloring[idx]
                new_color = random.randint(0, k - 1)
                
                if old_color == new_color:
                    continue
                
                # Calculate change in violations
                delta_violations = 0
                for ap_id in ap_indices[idx]:
                    ap = aps[ap_id]
                    c0 = coloring[ap[0]]
                    c1 = coloring[ap[1]]
                    c2 = coloring[ap[2]]
                    c3 = coloring[ap[3]]
                    
                    old_set_size = len({c0, c1, c2, c3})
                    
                    new_c0 = new_color if ap[0] == idx else c0
                    new_c1 = new_color if ap[1] == idx else c1
                    new_c2 = new_color if ap[2] == idx else c2
                    new_c3 = new_color if ap[3] == idx else c3
                    
                    new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                    
                    if old_set_size < 3 and new_set_size >= 3:
                        delta_violations -= 1
                    elif old_set_size >= 3 and new_set_size < 3:
                        delta_violations += 1
                
                if delta_violations < 0:
                    coloring[idx] = new_color
                    current_violations += delta_violations
                    if current_violations < best_local_violations:
                        best_local_violations = current_violations
                        best_local_coloring = coloring[:]
                else:
                    if delta_violations == 0:
                        accept = random.random() < 0.3
                    else:
                        if T > 0.01:
                            accept = random.random() < np.exp(-delta_violations / T)
                        else:
                            accept = False
                    
                    if accept:
                        coloring[idx] = new_color
                        current_violations += delta_violations
                
                T *= cooling
            
            # Check if we found a valid coloring
            if current_violations == 0:
                actual_colors = len(set(coloring))
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_coloring = coloring[:]
            if best_local_violations == 0:
                actual_colors = len(set(best_local_coloring))
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_coloring = best_local_coloring[:]
            
            # If we found a valid coloring with k colors, we can stop increasing k
            if best_coloring is not None and best_num_colors <= k:
                break
    
    # If we found a valid coloring, return it
    if best_coloring is not None:
        return best_coloring
    
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check 4-term APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 3 * d >= 1:
                    prev_colors = {colors[x - 1 - j * d] for j in (1, 2, 3)}
                    if len({color} | prev_colors) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(127)
    
    return colors
