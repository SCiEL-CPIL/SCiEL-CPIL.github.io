import time
import random
import numpy as np

N = 256
APPS_LIMIT = 150  # wall-clock limit in seconds

def _build_aps():
    """Build list of all 4-term arithmetic progressions in 0..N-1."""
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

def _build_pos_to_aps(aps):
    """Map each position to the list of AP indices it appears in."""
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
    return pos_to_aps

def _score_full(aps, colors):
    """Count monochromatic 4-APs. colors is a list or array of 0/1."""
    count = 0
    for a, b, c, d in aps:
        if colors[a] == colors[b] == colors[c] == colors[d]:
            count += 1
    return count

def _local_search_incremental(aps, pos_to_aps, colors, max_time=1.0):
    """
    Greedy local search with incremental delta computation.
    Flips a single position if it reduces the mono count.
    Returns (colors, score).
    """
    colors = list(colors)
    n_aps = len(aps)
    
    # Precompute count0 and count1 for each AP
    count0 = np.zeros(n_aps, dtype=np.int8)
    count1 = np.zeros(n_aps, dtype=np.int8)
    for k, (a, b, c, d) in enumerate(aps):
        n0 = (colors[a] == 0) + (colors[b] == 0) + (colors[c] == 0) + (colors[d] == 0)
        count0[k] = n0
        count1[k] = 4 - n0
    
    score = int(np.sum((count0 == 4) | (count1 == 4)))
    
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < max_time:
        improved = False
        order = list(range(N))
        random.shuffle(order)
        
        for i in order:
            old_val = colors[i]
            new_val = 1 - old_val
            affected = pos_to_aps[i]
            
            delta = 0
            # Phase 1: compute delta without modifying state
            for ap_idx in affected:
                was_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                if old_val == 0:
                    nc0 = count0[ap_idx] - 1
                    nc1 = count1[ap_idx] + 1
                else:
                    nc0 = count0[ap_idx] + 1
                    nc1 = count1[ap_idx] - 1
                is_mono = (nc0 == 4) or (nc1 == 4)
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            if delta < 0:
                colors[i] = new_val
                score += delta
                improved = True
                # Phase 2: update counts
                for ap_idx in affected:
                    if old_val == 0:
                        count0[ap_idx] -= 1
                        count1[ap_idx] += 1
                    else:
                        count0[ap_idx] += 1
                        count1[ap_idx] -= 1
    
    return colors, score

def _simulated_annealing(aps, pos_to_aps, colors, max_time=2.0):
    """
    Simulated annealing with 1-bit flips and occasional 3-bit flips.
    """
    colors = list(colors)
    n_aps = len(aps)
    
    count0 = np.zeros(n_aps, dtype=np.int8)
    count1 = np.zeros(n_aps, dtype=np.int8)
    for k, (a, b, c, d) in enumerate(aps):
        n0 = (colors[a] == 0) + (colors[b] == 0) + (colors[c] == 0) + (colors[d] == 0)
        count0[k] = n0
        count1[k] = 4 - n0
    
    score = int(np.sum((count0 == 4) | (count1 == 4)))
    best_score = score
    best_colors = colors[:]
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    
    while (time.time() - start) < max_time:
        T *= cooling
        if T < 0.01:
            break
        
        # Choose flip type: mostly 1-bit, sometimes 3-bit
        if random.random() < 0.1:
            # 3-bit flip
            indices = random.sample(range(N), 3)
            old_vals = [colors[i] for i in indices]
            new_vals = [1 - v for v in old_vals]
            
            delta = 0
            for i, nv in zip(indices, new_vals):
                old_val = colors[i]
                for ap_idx in pos_to_aps[i]:
                    was_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                    if old_val == 0:
                        nc0 = count0[ap_idx] - 1
                        nc1 = count1[ap_idx] + 1
                    else:
                        nc0 = count0[ap_idx] + 1
                        nc1 = count1[ap_idx] - 1
                    is_mono = (nc0 == 4) or (nc1 == 4)
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
            
            # Apply all three flips
            for i, nv in zip(indices, new_vals):
                colors[i] = nv
            for i, old_val in zip(indices, old_vals):
                for ap_idx in pos_to_aps[i]:
                    if old_val == 0:
                        count0[ap_idx] -= 1
                        count1[ap_idx] += 1
                    else:
                        count0[ap_idx] += 1
                        count1[ap_idx] -= 1
            
            new_score = score + delta
            if new_score < best_score:
                best_score = new_score
                best_colors = colors[:]
            
            if new_score < score or random.random() < np.exp(-delta / T):
                score = new_score
            else:
                # Revert
                for i, nv in zip(indices, new_vals):
                    colors[i] = old_vals[indices.index(i)]
                for i, old_val in zip(indices, old_vals):
                    for ap_idx in pos_to_aps[i]:
                        if old_val == 0:
                            count0[ap_idx] += 1
                            count1[ap_idx] -= 1
                        else:
                            count0[ap_idx] -= 1
                            count1[ap_idx] += 1
        else:
            # 1-bit flip
            i = random.randint(0, N - 1)
            old_val = colors[i]
            new_val = 1 - old_val
            affected = pos_to_aps[i]
            
            delta = 0
            for ap_idx in affected:
                was_mono = (count0[ap_idx] == 4) or (count1[ap_idx] == 4)
                if old_val == 0:
                    nc0 = count0[ap_idx] - 1
                    nc1 = count1[ap_idx] + 1
                else:
                    nc0 = count0[ap_idx] + 1
                    nc1 = count1[ap_idx] - 1
                is_mono = (nc0 == 4) or (nc1 == 4)
                if was_mono and not is_mono:
                    delta -= 1
                elif not was_mono and is_mono:
                    delta += 1
            
            colors[i] = new_val
            for ap_idx in affected:
                if old_val == 0:
                    count0[ap_idx] -= 1
                    count1[ap_idx] += 1
                else:
                    count0[ap_idx] += 1
                    count1[ap_idx] -= 1
            
            new_score = score + delta
            if new_score < best_score:
                best_score = new_score
                best_colors = colors[:]
            
            if new_score < score or random.random() < np.exp(-delta / T):
                score = new_score
            else:
                colors[i] = old_val
                for ap_idx in affected:
                    if old_val == 0:
                        count0[ap_idx] += 1
                        count1[ap_idx] -= 1
                    else:
                        count0[ap_idx] -= 1
                        count1[ap_idx] += 1
    
    return best_colors, best_score

def _ga_population_init(aps, pos_to_aps):
    """Generate a diverse initial population."""
    pop = []
    
    # Thue-Morse
    tm = [bin(i).count('1') % 2 for i in range(N)]
    pop.append(tm)
    
    # Random
    for _ in range(15):
        pop.append([random.randint(0, 1) for _ in range(N)])
    
    # Block patterns
    for bs in [2, 4, 8, 16, 32, 64]:
        pop.append([(i // bs) % 2 for i in range(N)])
    
    # Alternating blocks of varying sizes
    for bs in [4, 8, 16]:
        pattern = []
        for i in range(N):
            block_idx = i // bs
            pattern.append(block_idx % 2)
        pop.append(pattern)
    
    return pop

def construct():
    start_time = time.time()
    deadline = start_time + APPS_LIMIT - 5  # safety margin
    
    random.seed(42)
    aps = _build_aps()
    pos_to_aps = _build_pos_to_aps(aps)
    
    best_score = float('inf')
    best_colors = None
    
    # Phase 1: Local search on diverse starts
    population = _ga_population_init(aps, pos_to_aps)
    
    for cand in population:
        if time.time() > deadline - 30:
            break
        res_colors, res_score = _local_search_incremental(aps, pos_to_aps, cand, max_time=1.5)
        if res_score < best_score:
            best_score = res_score
            best_colors = res_colors[:]
    
    # Phase 2: Simulated annealing from best and some random starts
    sa_starts = [best_colors] if best_colors else []
    for _ in range(5):
        sa_starts.append([random.randint(0, 1) for _ in range(N)])
    
    for sa_start in sa_starts:
        if time.time() > deadline - 20:
            break
        sa_colors, sa_score = _simulated_annealing(aps, pos_to_aps, sa_start, max_time=3.0)
        if sa_score < best_score:
            best_score = sa_score
            best_colors = sa_colors[:]
    
    # Phase 3: Final local search polish
    if best_colors is not None and time.time() < deadline - 5:
        final_colors, final_score = _local_search_incremental(aps, pos_to_aps, best_colors, max_time=5.0)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors[:]
    
    if best_colors is None:
        best_colors = [random.randint(0, 1) for _ in range(N)]
    
    return best_colors
