import time
import numpy as np
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Precompute all 4-term APs
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                i0 = a - 1
                i1 = a + d - 1
                i2 = a + 2 * d - 1
                i3 = a + 3 * d - 1
                aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    
    # Precompute for each index, the APs it belongs to and the other indices
    # idx_info[i] = list of (ap_idx, o1, o2, o3)
    idx_info = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        i0, i1, i2, i3 = ap
        idx_info[i0].append((ap_idx, i1, i2, i3))
        idx_info[i1].append((ap_idx, i0, i2, i3))
        idx_info[i2].append((ap_idx, i0, i1, i3))
        idx_info[i3].append((ap_idx, i0, i1, i2))
        
    # Helper to count mono APs for a given coloring
    def count_mono(coloring):
        count = 0
        for ap in aps:
            c0 = coloring[ap[0]]
            if c0 == coloring[ap[1]] == coloring[ap[2]] == coloring[ap[3]]:
                count += 1
        return count

    # Helper to get initial mono status
    def get_mono_status(coloring):
        status = np.zeros(num_aps, dtype=bool)
        for ap_idx, ap in enumerate(aps):
            c0 = coloring[ap[0]]
            if c0 == coloring[ap[1]] == coloring[ap[2]] == coloring[ap[3]]:
                status[ap_idx] = True
        return status

    # Generate initial candidates
    seed_coloring = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        
    tm_coloring = thue_morse(N)
    rand_coloring = np.random.randint(0, 2, N, dtype=np.int8)
    
    candidates = [seed_coloring, tm_coloring, rand_coloring]
    
    best_coloring = None
    best_score = float('inf')
    best_status = None
    
    for c in candidates:
        score = count_mono(c)
        if score < best_score:
            best_score = score
            best_coloring = c.copy()
            best_status = get_mono_status(c)
            
    current_coloring = best_coloring.copy()
    current_status = best_status.copy()
    current_score = best_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    
    # Local variables for speed
    idx_info_local = idx_info
    current_coloring_local = current_coloring
    current_status_local = current_status
    current_score_local = current_score
    best_score_local = best_score
    best_coloring_local = best_coloring
    best_status_local = best_status
    
    while time.time() < deadline:
        idx = np.random.randint(N)
        new_color = 1 - current_coloring_local[idx]
        
        delta = 0
        for ap_idx, o1, o2, o3 in idx_info_local[idx]:
            is_mono = current_status_local[ap_idx]
            if is_mono:
                delta -= 1
            else:
                if current_coloring_local[o1] == new_color and \
                   current_coloring_local[o2] == new_color and \
                   current_coloring_local[o3] == new_color:
                    delta += 1
        
        new_score = current_score_local + delta
        
        if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            current_coloring_local[idx] = new_color
            current_score_local = new_score
            
            for ap_idx, o1, o2, o3 in idx_info_local[idx]:
                c_idx = current_coloring_local[idx]
                if c_idx == current_coloring_local[o1] == current_coloring_local[o2] == current_coloring_local[o3]:
                    current_status_local[ap_idx] = True
                else:
                    current_status_local[ap_idx] = False
            
            if current_score_local < best_score_local:
                best_score_local = current_score_local
                best_coloring_local = current_coloring_local.copy()
                best_status_local = current_status_local.copy()
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final Polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(N):
            new_color = 1 - current_coloring_local[idx]
            delta = 0
            for ap_idx, o1, o2, o3 in idx_info_local[idx]:
                is_mono = current_status_local[ap_idx]
                if is_mono:
                    delta -= 1
                else:
                    if current_coloring_local[o1] == new_color and \
                       current_coloring_local[o2] == new_color and \
                       current_coloring_local[o3] == new_color:
                        delta += 1
            
            if delta < 0:
                current_coloring_local[idx] = new_color
                current_score_local += delta
                
                for ap_idx, o1, o2, o3 in idx_info_local[idx]:
                    c_idx = current_coloring_local[idx]
                    if c_idx == current_coloring_local[o1] == current_coloring_local[o2] == current_coloring_local[o3]:
                        current_status_local[ap_idx] = True
                    else:
                        current_status_local[ap_idx] = False
                
                if current_score_local < best_score_local:
                    best_score_local = current_score_local
                    best_coloring_local = current_coloring_local.copy()
                    best_status_local = current_status_local.copy()
                
                improved = True
                break
        if improved:
            continue
        else:
            break

    return best_coloring_local.tolist()
