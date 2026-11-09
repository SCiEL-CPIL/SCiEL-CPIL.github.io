import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
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
    
    # Precompute "others" for each position to speed up flip_delta
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]  # shape (K, 4)
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)
    
    # Count monochromatic APs for a coloring (0/1 array)
    def count_mono(color):
        cols = color[aps_arr]  # shape (M, 4)
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    # Compute delta in mono count if we flip position p
    def flip_delta(color, p):
        others = pos_others[p]
        if len(others) == 0:
            return 0
        other_colors = color[others]  # shape (K, 3)
        cp = color[p]
        new_cp = 1 - cp
        all_same_now = np.all(other_colors == cp, axis=1)
        all_same_new = np.all(other_colors == new_cp, axis=1)
        delta = np.sum((~all_same_now) & all_same_new) - np.sum(all_same_now & (~all_same_new))
        return int(delta)
    
    # Vectorized flip_delta for multiple positions at once
    def flip_deltas_batch(color, positions):
        """Compute flip deltas for a batch of positions."""
        deltas = np.zeros(len(positions), dtype=np.int32)
        for i, p in enumerate(positions):
            deltas[i] = flip_delta(color, p)
        return deltas
    
    # Thue-Morse initial coloring
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
    
    # Vector Annealing: evolve a population of spin configurations
    # Population P of size pop_size, each of length n
    # Energy = count of monochromatic 4-APs
    def vector_annealing(population, max_iter=2000, start_temp=5.0, end_temp=0.01):
        pop_size = population.shape[0]
        current_scores = np.array([count_mono(population[i]) for i in range(pop_size)], dtype=np.int32)
        best_scores = current_scores.copy()
        best_population = population.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Temperature schedule: geometric cooling
            ratio = it / max_iter
            temp = start_temp * (end_temp / start_temp) ** ratio
            
            # For each member in population, propose a random flip
            for i in range(pop_size):
                p = np.random.randint(0, n)
                delta = flip_delta(population[i], p)
                
                # Metropolis acceptance
                if delta < 0 or np.random.random() < np.exp(-delta / (temp + 1e-10)):
                    population[i, p] = 1 - population[i, p]
                    current_scores[i] += delta
                    
                    if current_scores[i] < best_scores[i]:
                        best_scores[i] = current_scores[i]
                        best_population[i] = population[i].copy()
            
            # Elite preservation: replace worst with best (with some mutation)
            worst_idx = np.argmax(current_scores)
            best_idx = np.argmin(best_scores)
            if best_scores[best_idx] < current_scores[worst_idx]:
                population[worst_idx] = best_population[best_idx].copy()
                # Add small perturbation
                num_flips = np.random.randint(1, 4)
                flip_pos = np.random.choice(n, size=num_flips, replace=False)
                population[worst_idx, flip_pos] = 1 - population[worst_idx, flip_pos]
                current_scores[worst_idx] = count_mono(population[worst_idx])
        
        # Return the best individual
        best_overall_idx = np.argmin(best_scores)
        return best_population[best_overall_idx], int(best_scores[best_overall_idx])
    
    # Local search: hill climbing with best-improvement
    def local_search(color, max_iter=5000):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            best_delta = 0
            best_pos = -1
            for p in range(n):
                d = flip_delta(color, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                current_score += best_delta
                improved = True
            if not improved:
                break
        return color, current_score
    
    # Iterated local search with Vector Annealing
    def iterated_local_search(color, max_iterations=100):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline - 5:
                break
            
            # Phase 1: Create a small population from current best + perturbations
            pop_size = 16
            population = np.zeros((pop_size, n), dtype=np.int8)
            population[0] = color
            
            for j in range(1, pop_size):
                population[j] = color.copy()
                # Random perturbation: flip 1-5 random positions
                num_flips = np.random.randint(1, 6)
                flip_pos = np.random.choice(n, size=num_flips, replace=False)
                population[j, flip_pos] = 1 - population[j, flip_pos]
            
            # Phase 2: Vector Annealing on the population
            best_color_va, best_score_va = vector_annealing(population, max_iter=500)
            
            # Phase 3: Local search to polish
            best_color_ls, best_score_ls = local_search(best_color_va, max_iter=2000)
            
            if best_score_ls < best_score:
                best_score = best_score_ls
                best_color = best_color_ls.copy()
            
            # Update color for next iteration
            color = best_color.copy()
        
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(8):
        startings.append(random_coloring())
    
    # Also try Thue-Morse with random perturbations
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    for start_color in startings:
        if time.time() - start_time > deadline - 10:
            break
        
        # Run iterated local search
        color, score = iterated_local_search(start_color, max_iterations=50)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # If we have time, do more iterations on the best
    if time.time() - start_time < deadline - 20 and best_overall_color is not None:
        color, score = iterated_local_search(best_overall_color, max_iterations=200)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
