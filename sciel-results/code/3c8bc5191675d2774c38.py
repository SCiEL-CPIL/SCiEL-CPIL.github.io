import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs: (i0, i1, i2, i3) 0-indexed
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    # Count monochromatic APs for a coloring (0/1 array)
    def count_mono(color):
        cols = color[aps_arr]  # shape (M, 4)
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    # Compute delta in mono count if we flip position p
    def flip_delta(color, p):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            return 0
        delta = 0
        new_color_p = 1 - color[p]
        for ap_idx in aps_at_p:
            i0, i1, i2, i3 = aps_arr[ap_idx]
            others = [i for i in (i0, i1, i2, i3) if i != p]
            c0, c1, c2 = color[others[0]], color[others[1]], color[others[2]]
            all_same_now = (c0 == c1 == c2 == color[p])
            all_same_new = (c0 == c1 == c2 == new_color_p)
            if not all_same_now and all_same_new:
                delta += 1
            elif all_same_now and not all_same_new:
                delta -= 1
        return delta
    
    # Thue-Morse initial coloring
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    # Generate several diverse starting points
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color
    
    # Simulated annealing with efficient delta evaluation
    def simulated_annealing(color, T_start=10.0, T_end=0.01, max_iter=50000):
        color = color.copy()
        current_score = count_mono(color)
        best_color = color.copy()
        best_score = current_score
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Update temperature
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            # Pick random position to flip
            p = np.random.randint(0, n)
            d = flip_delta(color, p)
            
            # Accept if improvement, or with probability exp(-d/T) if worsening
            if d < 0 or (d > 0 and np.random.random() < np.exp(-d / T)):
                color[p] = 1 - color[p]
                current_score += d
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
        
        return best_color, best_score
    
    # Local search: hill climbing with best-improvement
    def local_search_best(color, max_iter=100):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            best_p = -1
            best_d = 0
            for p in range(n):
                d = flip_delta(color, p)
                if d < best_d:
                    best_d = d
                    best_p = p
            if best_p >= 0:
                color[best_p] = 1 - color[best_p]
                current_score += best_d
                improved = True
        return color, current_score
    
    # Iterated local search with simulated annealing perturbation
    def iterated_search(color, max_iterations=30):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline - 5:
                break
            
            # Local search
            color, score = local_search_best(color, max_iter=50)
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
            
            # Perturbation: flip random subset of positions
            num_flips = np.random.randint(5, 20)
            flip_positions = np.random.choice(n, size=num_flips, replace=False)
            color[flip_positions] = 1 - color[flip_positions]
        
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(10):
        startings.append(random_coloring())
    
    # Also try Thue-Morse with random perturbations
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # First pass: run iterated search on diverse starting points
    for start_color in startings:
        if time.time() - start_time > deadline - 15:
            break
        
        color, score = iterated_search(start_color, max_iterations=20)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # Second pass: simulated annealing from best found
    if time.time() - start_time < deadline - 20 and best_overall_color is not None:
        color, score = simulated_annealing(best_overall_color, T_start=5.0, T_end=0.01, max_iter=30000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
        
        # Another round of local search polish
        if time.time() - start_time < deadline - 10:
            color, score = local_search_best(best_overall_color, max_iter=200)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
    
    # Final polish: multiple rounds of local search
    if time.time() - start_time < deadline - 5 and best_overall_color is not None:
        for _ in range(3):
            if time.time() - start_time > deadline - 5:
                break
            color, score = local_search_best(best_overall_color, max_iter=100)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
