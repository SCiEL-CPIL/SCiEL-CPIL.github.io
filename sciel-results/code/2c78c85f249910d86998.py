import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0
    n = 256

    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    aps_arr = np.array(aps, dtype=np.int32)
    
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)

    def count_mono(color):
        cols = color[aps_arr]
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    def flip_delta(color, p):
        others = pos_others[p]
        if others.shape[0] == 0:
            return 0
        c_p = color[p]
        c_new = 1 - c_p
        others_colors = color[others]
        is_mono_now = np.all(others_colors == c_p, axis=1)
        is_mono_new = np.all(others_colors == c_new, axis=1)
        return int(np.sum(is_mono_new) - np.sum(is_mono_now))
    
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color

    def simulated_annealing(color):
        color = color.copy()
        current_score = count_mono(color)
        best_color = color.copy()
        best_score = current_score
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.995
        
        T = T_start
        
        while time.time() - start_time < deadline - 2 and T > T_end:
            p = np.random.randint(n)
            delta = flip_delta(color, p)
            
            if delta < 0:
                accept = True
            else:
                accept = (np.random.rand() < np.exp(-delta / T))
            
            if accept:
                color[p] = 1 - color[p]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            
            T *= cooling_rate
            
        return best_color, best_score

    startings = [thue_morse(n)]
    for bs in [2, 4, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(5):
        startings.append(random_coloring())
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=15, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)

    scored_startings = []
    for sc in startings:
        sc_score = count_mono(sc)
        scored_startings.append((sc_score, sc))
    scored_startings.sort(key=lambda x: x[0])
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    runs_per_start = 50
    
    for sc_score, sc in scored_startings:
        for _ in range(runs_per_start):
            if time.time() - start_time > deadline - 5:
                break
            color, score = simulated_annealing(sc)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
        
        if time.time() - start_time > deadline - 5:
            break

    if time.time() - start_time < deadline - 10 and best_overall_color is not None:
        while time.time() - start_time < deadline - 2:
            color, score = simulated_annealing(best_overall_color)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()

    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
