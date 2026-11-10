"""Optimized 2-colouring of [1,256] to minimize monochromatic 4-term APs."""
import time
import numpy as np

N = 256
TIME_LIMIT = 140.0  # safety margin before 150s kill
SEED = 42


def _build_aps():
    """Build all 4-term arithmetic progressions as index tuples (0-indexed)."""
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps


def _build_aps_array(aps):
    """Convert AP list to numpy array for vectorized counting."""
    return np.array(aps, dtype=np.int32)


def _count_mono_vec(aps_arr, colors):
    """Count monochromatic 4-APs using vectorized numpy ops."""
    c0 = colors[aps_arr[:, 0]]
    c1 = colors[aps_arr[:, 1]]
    c2 = colors[aps_arr[:, 2]]
    c3 = colors[aps_arr[:, 3]]
    return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))


def _count_mono_fast(aps, colors):
    """Count monochromatic 4-APs with Python loop."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count


def _build_ap_index(aps_arr):
    """Build an index mapping each position to the list of AP indices it belongs to.
    Also build for each AP index, the other 3 positions."""
    n_aps = len(aps_arr)
    pos_to_aps = [[] for _ in range(N)]
    for idx in range(n_aps):
        for pos in aps_arr[idx]:
            pos_to_aps[pos].append(idx)
    return pos_to_aps


def _count_mono_delta(pos_to_aps, aps_arr, colors, flip_pos):
    """Compute the change in mono-AP count if we flip position flip_pos.
    Returns delta = new_count - old_count (negative means improvement)."""
    old_count = 0
    new_count = 0
    for ap_idx in pos_to_aps[flip_pos]:
        ap = aps_arr[ap_idx]
        # Check if this AP is currently monochromatic
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        is_mono_old = (c0 == c1 == c2 == c3)
        
        # After flip, check if still monochromatic
        new_c = 1 - colors[flip_pos]
        # Rebuild the 4 colors with flip
        nc0 = new_c if ap[0] == flip_pos else c0
        nc1 = new_c if ap[1] == flip_pos else c1
        nc2 = new_c if ap[2] == flip_pos else c2
        nc3 = new_c if ap[3] == flip_pos else c3
        is_mono_new = (nc0 == nc1 == nc2 == nc3)
        
        if is_mono_old:
            old_count += 1
        if is_mono_new:
            new_count += 1
    
    return new_count - old_count


def _local_search_delta(pos_to_aps, aps_arr, colors, max_passes=20, deadline=None):
    """Local search using delta evaluation for speed. Returns improved colors and count."""
    colors = colors.copy()
    current_count = _count_mono_vec(aps_arr, colors)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        order = np.random.permutation(N)
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            delta = _count_mono_delta(pos_to_aps, aps_arr, colors, i)
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_count += delta
                improved = True
        if not improved:
            break
    
    return colors, current_count


def _simulated_annealing_delta(pos_to_aps, aps_arr, colors, start_temp=5.0, end_temp=0.1,
                                max_iter=10000, deadline=None):
    """Simulated annealing with delta evaluation."""
    colors = colors.copy()
    current_count = _count_mono_vec(aps_arr, colors)
    best_colors = colors.copy()
    best_count = current_count
    T = start_temp
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        i = np.random.randint(N)
        delta = _count_mono_delta(pos_to_aps, aps_arr, colors, i)
        
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / max(T, 1e-9))):
            colors[i] = 1 - colors[i]
            current_count += delta
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()
        else:
            pass  # don't flip
        
        # Cool down
        T = start_temp * (end_temp / start_temp) ** (it / max_iter)
    
    return best_colors, best_count


def _random_initial():
    """Random binary coloring."""
    return np.random.randint(0, 2, N).astype(np.int32)


def _structured_inits():
    """Generate diverse structured initial colorings."""
    inits = []
    # Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
    inits.append(tm)
    # Block patterns of various sizes
    for block in [2, 3, 4, 5, 6, 8, 10, 12, 16]:
        c = np.array([(i // block) % 2 for i in range(N)], dtype=np.int32)
        inits.append(c)
    # Alternating
    alt = np.array([i % 2 for i in range(N)], dtype=np.int32)
    inits.append(alt)
    # Random with bias
    for bias in [0.3, 0.4, 0.6, 0.7]:
        c = (np.random.random(N) < bias).astype(np.int32)
        inits.append(c)
    return inits


def construct():
    np.random.seed(SEED)
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Precompute APs
    aps_list = _build_aps()
    aps_arr = _build_aps_array(aps_list)
    pos_to_aps = _build_ap_index(aps_arr)
    
    # Initialize population
    population = []
    scores = []
    
    # Add structured inits
    for c in _structured_inits():
        population.append(c)
        scores.append(_count_mono_vec(aps_arr, c))
    
    # Add random inits
    for _ in range(15):
        c = _random_initial()
        population.append(c)
        scores.append(_count_mono_vec(aps_arr, c))
    
    best_score = min(scores)
    best_colors = population[scores.index(best_score)].copy()
    
    # Main search loop
    gen = 0
    while time.time() < deadline - 20:
        gen += 1
        
        # Select best individual and run local search
        idx_best = np.argmin(scores)
        c_best = population[idx_best].copy()
        
        # Local search with delta
        c_ls, s_ls = _local_search_delta(pos_to_aps, aps_arr, c_best, 
                                          max_passes=30, deadline=deadline - 15)
        if s_ls < best_score:
            best_score = s_ls
            best_colors = c_ls.copy()
            population[idx_best] = c_ls
            scores[idx_best] = s_ls
        
        # Simulated annealing on best
        if time.time() < deadline - 25:
            c_sa, s_sa = _simulated_annealing_delta(pos_to_aps, aps_arr, best_colors,
                                                     start_temp=3.0, end_temp=0.05,
                                                     max_iter=5000, deadline=deadline - 15)
            if s_sa < best_score:
                best_score = s_sa
                best_colors = c_sa.copy()
        
        # Generate new candidates via mutation of best
        if time.time() < deadline - 30:
            for _ in range(5):
                if time.time() > deadline - 25:
                    break
                c_mut = best_colors.copy()
                mask = np.random.random(N) < 0.05
                c_mut[mask] = 1 - c_mut[mask]
                c_ls2, s_ls2 = _local_search_delta(pos_to_aps, aps_arr, c_mut,
                                                    max_passes=20, deadline=deadline - 15)
                if s_ls2 < best_score:
                    best_score = s_ls2
                    best_colors = c_ls2.copy()
                
                # Add to population if good
                if len(population) < 30:
                    population.append(c_ls2)
                    scores.append(s_ls2)
                else:
                    # Replace worst
                    worst_idx = np.argmax(scores)
                    if s_ls2 < scores[worst_idx]:
                        population[worst_idx] = c_ls2
                        scores[worst_idx] = s_ls2
        
        # Periodic full restart with new random init
        if gen % 3 == 0 and time.time() < deadline - 30:
            c_new = _random_initial()
            c_ls3, s_ls3 = _local_search_delta(pos_to_aps, aps_arr, c_new,
                                                max_passes=25, deadline=deadline - 15)
            if s_ls3 < best_score:
                best_score = s_ls3
                best_colors = c_ls3.copy()
            
            if len(population) < 30:
                population.append(c_ls3)
                scores.append(s_ls3)
            else:
                worst_idx = np.argmax(scores)
                if s_ls3 < scores[worst_idx]:
                    population[worst_idx] = c_ls3
                    scores[worst_idx] = s_ls3
    
    # Final polish
    if time.time() < deadline - 2:
        c_final, s_final = _local_search_delta(pos_to_aps, aps_arr, best_colors,
                                                max_passes=50, deadline=deadline - 1)
        if s_final < best_score:
            best_score = s_final
            best_colors = c_final
    
    # One more SA pass if time
    if time.time() < deadline - 3:
        c_sa2, s_sa2 = _simulated_annealing_delta(pos_to_aps, aps_arr, best_colors,
                                                   start_temp=2.0, end_temp=0.01,
                                                   max_iter=3000, deadline=deadline - 1)
        if s_sa2 < best_score:
            best_score = s_sa2
            best_colors = c_sa2
    
    return [int(x) for x in best_colors]
