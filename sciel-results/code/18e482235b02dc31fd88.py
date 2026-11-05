import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 128
    # Precompute 4-APs
    aps = []
    aps_containing = [[] for _ in range(N)]
    aps_ending_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            ap_idx = len(aps)
            ap = (i0, i1, i2, i3)
            aps.append(ap)
            for pos in ap:
                aps_containing[pos].append(ap_idx)
            aps_ending_at[i3].append(ap_idx)
            
    n_aps = len(aps)
    
    best_colors = None
    best_distinct = N + 1
    
    # Try k from 3 to 7
    for k in range(3, 8):
        if time.time() > deadline:
            break
            
        for restart in range(100):
            if time.time() > deadline:
                break
                
            # Greedy Initialization
            colors = np.zeros(N, dtype=np.uint8)
            masks = np.zeros(N, dtype=np.int64)
            
            for i in range(N):
                candidate_colors = list(range(k))
                random.shuffle(candidate_colors)
                
                placed = False
                for c in candidate_colors:
                    valid = True
                    # Check APs ending at i
                    for ap_idx in aps_ending_at[i]:
                        i0, i1, i2, i3 = aps[ap_idx]
                        m = masks[i0] | masks[i1] | masks[i2] | (1 << c)
                        if m.bit_count() < 3:
                            valid = False
                            break
                    
                    if valid:
                        colors[i] = c
                        masks[i] = 1 << c
                        placed = True
                        break
                
                if not placed:
                    c = random.randint(0, k-1)
                    colors[i] = c
                    masks[i] = 1 << c
            
            # Calculate initial violations
            violations = np.zeros(n_aps, dtype=np.int8)
            current_violations = 0
            for ap_idx in range(n_aps):
                i0, i1, i2, i3 = aps[ap_idx]
                m = masks[i0] | masks[i1] | masks[i2] | masks[i3]
                if m.bit_count() < 3:
                    violations[ap_idx] = 1
                    current_violations += 1
            
            best_local_violations = current_violations
            best_local_colors = colors.copy()
            
            # Local Search
            max_steps = 10000
            temp = 1.0
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta = 0
                affected_aps = aps_containing[pos]
                
                for ap_idx in affected_aps:
                    i0, i1, i2, i3 = aps[ap_idx]
                    if i0 == pos:
                        nm = (1 << new_color) | masks[i1] | masks[i2] | masks[i3]
                    elif i1 == pos:
                        nm = masks[i0] | (1 << new_color) | masks[i2] | masks[i3]
                    elif i2 == pos:
                        nm = masks[i0] | masks[i1] | (1 << new_color) | masks[i3]
                    else:
                        nm = masks[i0] | masks[i1] | masks[i2] | (1 << new_color)
                    
                    new_inv = 0 if nm.bit_count() >= 3 else 1
                    old_inv = violations[ap_idx]
                    delta += new_inv - old_inv
                    
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.5
                else:
                    if random.random() < math.exp(-delta / temp):
                        accept = True

                if accept:
                    colors[pos] = new_color
                    masks[pos] = 1 << new_color
                    current_violations += delta
                    
                    for ap_idx in affected_aps:
                        i0, i1, i2, i3 = aps[ap_idx]
                        m = masks[i0] | masks[i1] | masks[i2] | masks[i3]
                        violations[ap_idx] = 0 if m.bit_count() >= 3 else 1
                        
                    if current_violations < best_local_violations:
                        best_local_violations = current_violations
                        best_local_colors = colors.copy()
                        
                        if best_local_violations == 0:
                            distinct = len(set(best_local_colors.tolist()))
                            if distinct < best_distinct:
                                best_distinct = distinct
                                best_colors = best_local_colors.copy()
                            break
                            
                temp *= 0.995
                
                if best_local_violations == 0:
                    break
                    
            if best_local_violations == 0:
                distinct = len(set(best_local_colors.tolist()))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = best_local_colors.copy()
                    
                if k == 3:
                    return best_colors.tolist()
                
    if best_colors is not None:
        return best_colors.tolist()
    else:
        return list(range(128))
