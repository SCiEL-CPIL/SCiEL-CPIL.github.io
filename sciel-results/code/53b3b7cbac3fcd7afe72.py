"""Two-colourings of [1,256] with few monochromatic 4-term APs."""
import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a, d >= 1, a+3d <= n
    # Using 0-indexed: positions 0..n-1 correspond to integers 1..n
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, store list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps_arr[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Vectorized delta computation structures
    pos_ap_indices = []
    pos_ap_offsets = []
    for i in range(n):
        if pos_aps[i]:
            indices, offsets = zip(*pos_aps[i])
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int8))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int8))
    
    # Count monochromatic 4-APs given a coloring (array of 0/1) - vectorized
    def count_mono(color):
        c = color[aps_arr]
        mono = (c[:, 1] == c[:, 0]) & (c[:, 2] == c[:, 0]) & (c[:, 3] == c[:, 0])
        return int(np.sum(mono))
    
    # Vectorized delta flip
    def delta_flip(color, pos):
        ap_idx = pos_ap_indices[pos]
        if len(ap_idx) == 0:
            return 0
        sub = color[aps_arr[ap_idx]]
        off = pos_ap_offsets[pos]
        curr_mono = (sub[:, 1] == sub[:, 0]) & (sub[:, 2] == sub[:, 0]) & (sub[:, 3] == sub[:, 0])
        curr_count = int(np.sum(curr_mono))
        new_sub = sub.copy()
        new_sub[np.arange(len(ap_idx)), off] = 1 - new_sub[np.arange(len(ap_idx)), off]
        new_mono = (new_sub[:, 1] == new_sub[:, 0]) & (new_sub[:, 2] == new_sub[:, 0]) & (new_sub[:, 3] == new_sub[:, 0])
        new_count = int(np.sum(new_mono))
        return new_count - curr_count
    
    # Thue-Morse initialization
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    # Initialize with Thue-Morse
    best_color = np.array(thue_morse(n), dtype=np.int8)
    best_score = count_mono(best_color)
    
    # Greedy local search with vectorized delta
    def greedy_local_search(color, score, time_budget=3.0):
        color = color.copy()
        improved = True
        start = time.time()
        while improved and time.time() - start < time_budget:
            improved = False
            for pos in range(n):
                if time.time() > deadline:
                    return color, score
                delta = delta_flip(color, pos)
                if delta < 0:
                    color[pos] = 1 - color[pos]
                    score += delta
                    improved = True
        return color, score
    
    # Simulated annealing with vectorized delta
    def simulated_annealing(color, score, time_budget=15.0):
        color = color.copy()
        start = time.time()
        T_start = 2.5
        T_end = 0.01
        best_score = score
        best_color = color.copy()
        
        while time.time() - start < time_budget and time.time() < deadline:
            elapsed = time.time() - start
            remaining_frac = max(0.0, (time_budget - elapsed) / time_budget)
            T = T_end + (T_start - T_end) * remaining_frac ** 1.3
            
            pos = np.random.randint(0, n)
            delta = delta_flip(color, pos)
            
            if delta <= 0:
                color[pos] = 1 - color[pos]
                score += delta
            else:
                accept_prob = np.exp(-delta / max(T, 0.001))
                if np.random.random() < accept_prob:
                    color[pos] = 1 - color[pos]
                    score += delta
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
        
        return best_color, best_score
    
    # Run initial local search from Thue-Morse
    best_color, best_score = greedy_local_search(best_color, best_score, time_budget=5.0)
    
    # Multi-start with diverse initializations
    random.seed(42)
    np.random.seed(42)
    
    num_starts = 40
    for start_iter in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Diverse initialization strategies
        if start_iter < 5:
            # Perturbed best
            color = best_color.copy()
            for _ in range(random.randint(5, 15)):
                color[random.randint(0, n - 1)] = 1 - color[random.randint(0, n - 1)]
        elif start_iter < 15:
            # Random
            color = np.random.randint(0, 2, n, dtype=np.int8)
        elif start_iter < 25:
            # Thue-Morse with perturbations
            color = np.array(thue_morse(n), dtype=np.int8)
            for _ in range(random.randint(10, 30)):
                color[random.randint(0, n - 1)] = 1 - color[random.randint(0, n - 1)]
        else:
            # Structured: blocks of alternating colors
            block_size = random.choice([4, 8, 16, 32])
            color = np.zeros(n, dtype=np.int8)
            for i in range(0, n, block_size * 2):
                color[i:i+block_size] = 0
                color[i+block_size:i+2*block_size] = 1
            # Add some random noise
            for _ in range(random.randint(5, 15)):
                color[random.randint(0, n - 1)] = 1 - color[random.randint(0, n - 1)]
        
        score = count_mono(color)
        
        # Greedy local search first
        color, score = greedy_local_search(color, score, time_budget=2.0)
        
        # Then simulated annealing
        color, score = simulated_annealing(color, score, time_budget=10.0)
        
        if score < best_score:
            best_score = score
            best_color = color.copy()
    
    # Final polish: more iterations of local search on the best
    best_color, best_score = greedy_local_search(best_color, best_score, time_budget=10.0)
    best_color, best_score = simulated_annealing(best_color, best_score, time_budget=15.0)
    best_color, best_score = greedy_local_search(best_color, best_score, time_budget=5.0)
    
    # Return as list of ints
    return [int(c) for c in best_color]
