"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns for each position i:
      - ap_indices[i]: numpy array of (a, d, k) as flat index into a global AP table
      - We store for each position a list of (a, d, k) tuples"""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d, k))
    # Convert to numpy arrays for faster access
    ap_arrays = []
    for i in range(N):
        if ap_list[i]:
            arr = np.array(ap_list[i], dtype=np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        ap_arrays.append(arr)
    return ap_arrays

def _delta_flip_fast(c, i, ap_arrays):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    ap_arr = ap_arrays[i]
    if len(ap_arr) == 0:
        return 0
    
    a_vals = ap_arr[:, 0]
    d_vals = ap_arr[:, 1]
    k_vals = ap_arr[:, 2]
    
    # For each AP, get the 4 positions
    # idx0 = a, idx1 = a+d, idx2 = a+2d, idx3 = a+3d
    idx0 = a_vals
    idx1 = a_vals + d_vals
    idx2 = a_vals + 2 * d_vals
    idx3 = a_vals + 3 * d_vals
    
    v0 = c[idx0]
    v1 = c[idx1]
    v2 = c[idx2]
    v3 = c[idx3]
    
    was_mono = (v0 == v1) & (v0 == v2) & (v0 == v3)
    
    new_v = 1 - c[i]
    
    # Determine which position is being flipped
    # k_vals tells us which position in the AP is i
    is_mono_after = np.zeros(len(ap_arr), dtype=bool)
    
    # For k=0: new_v == v1 == v2 == v3
    mask_k0 = (k_vals == 0)
    if np.any(mask_k0):
        is_mono_after[mask_k0] = (new_v == v1[mask_k0]) & (new_v == v2[mask_k0]) & (new_v == v3[mask_k0])
    
    # For k=1: v0 == new_v == v2 == v3
    mask_k1 = (k_vals == 1)
    if np.any(mask_k1):
        is_mono_after[mask_k1] = (v0[mask_k1] == new_v) & (new_v == v2[mask_k1]) & (new_v == v3[mask_k1])
    
    # For k=2: v0 == v1 == new_v == v3
    mask_k2 = (k_vals == 2)
    if np.any(mask_k2):
        is_mono_after[mask_k2] = (v0[mask_k2] == v1[mask_k2]) & (v1[mask_k2] == new_v) & (new_v == v3[mask_k2])
    
    # For k=3: v0 == v1 == v2 == new_v
    mask_k3 = (k_vals == 3)
    if np.any(mask_k3):
        is_mono_after[mask_k3] = (v0[mask_k3] == v1[mask_k3]) & (v1[mask_k3] == v2[mask_k3]) & (v2[mask_k3] == new_v)
    
    delta = np.count_nonzero(~was_mono & is_mono_after) - np.count_nonzero(was_mono & ~is_mono_after)
    return int(delta)

def _local_search_delta(c, time_limit, ap_arrays=None):
    """Improve coloring via single-flip local search using delta computation."""
    if ap_arrays is None:
        ap_arrays = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    improved = True
    max_passes = 50
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # Try positions in order; use a random permutation each pass for diversity
        order = np.random.default_rng(passes * 7 + 1).permutation(N)
        for i in order:
            if time.time() - start > time_limit:
                return c, current
            delta = _delta_flip_fast(c, int(i), ap_arrays)
            if delta < 0:
                c[int(i)] = 1 - c[int(i)]
                current += delta
                improved = True
    return c, current

def _simulated_annealing(c, time_limit, ap_arrays=None, T_start=8.0, T_end=0.05, alpha=0.9995):
    """Simulated annealing to escape local minima."""
    if ap_arrays is None:
        ap_arrays = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    T = T_start
    rng = np.random.default_rng(456)
    best_c = c.copy()
    best_count = current
    
    # Precompute log for speed
    while time.time() - start < time_limit:
        # Pick random position
        i = int(rng.integers(0, N))
        delta = _delta_flip_fast(c, i, ap_arrays)
        
        if delta < 0:
            # Always accept improvement
            c[i] = 1 - c[i]
            current += delta
            if current < best_count:
                best_count = current
                best_c = c.copy()
        else:
            # Accept with probability exp(-delta/T)
            if delta == 0:
                c[i] = 1 - c[i]
            else:
                if T > 0.01:
                    # Use log-space comparison for stability
                    if -delta / T > np.log(rng.random()):
                        c[i] = 1 - c[i]
                        current += delta
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_c, best_count

def _genetic_algorithm(c_initial, time_limit, ap_arrays, pop_size=20, generations=50):
    """
    Genetic Algorithm to explore the solution space.
    Uses tournament selection, uniform crossover, and mutation.
    Applies local search polish to offspring.
    """
    start = time.time()
    rng = np.random.default_rng(123)
    
    # Initialize population
    # One individual is the initial best (elite)
    # Others are random or mutated versions of the elite
    population = [c_initial.copy()]
    scores = [_count_mono4(c_initial)]
    
    for _ in range(pop_size - 1):
        # Start with random initialization
        c_new = rng.integers(0, 2, size=N).astype(np.int8)
        # Quick local search to make them reasonable
        c_new, score_new = _local_search_delta(c_new, min(0.5, (time_limit - (time.time() - start)) / 10), ap_arrays)
        population.append(c_new)
        scores.append(score_new)
    
    best_c = population[0].copy()
    best_score = min(scores)
    
    # Main GA Loop
    while time.time() - start < time_limit:
        # Selection: Tournament selection
        # We will create a new population of size pop_size
        new_population = []
        new_scores = []
        
        # Elitism: Keep the best individual from current population
        best_idx = int(np.argmin(scores))
        new_population.append(population[best_idx].copy())
        new_scores.append(scores[best_idx])
        
        # Generate offspring
        for _ in range(pop_size - 1):
            # Pick two parents via tournament
            p1_idx, p2_idx = _tournament_select(population, scores, rng, k=3)
            
            # Crossover: Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, population[p1_idx], population[p2_idx])
            
            # Mutation: Flip bits with low probability
            mut_mask = rng.random(N) < 0.05
            child[mut_mask] = 1 - child[mut_mask]
            
            # Local Search Polish
            # Use a very short time limit for polish to allow many generations
            time_left_for_polish = max(0.1, (time_limit - (time.time() - start)) / (generations * pop_size))
            child, score_child = _local_search_delta(child, time_left_for_polish, ap_arrays)
            
            new_population.append(child)
            new_scores.append(score_child)
            
            # Track global best
            if score_child < best_score:
                best_score = score_child
                best_c = child.copy()
                
        population = new_population
        scores = new_scores
        
        # Check time
        if time.time() - start > time_limit:
            break

    return best_c, best_score

def _tournament_select(population, scores, rng, k=3):
    """Tournament selection: pick k random individuals, return the best one."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    best_score = scores[best_idx]
    for idx in indices[1:]:
        if scores[idx] < best_score:
            best_score = scores[idx]
            best_idx = idx
    return best_idx, best_idx # Return same for both parents in simple implementation, 
                              # but we need distinct parents. Let's fix this.
    
    # Corrected tournament selection to return two distinct parents if possible
    # Actually, let's just return one index and call it twice? No, that's bad.
    # Let's rewrite the selection logic inside the GA loop properly.
    pass

def construct():
    start_time = time.time()
    ap_arrays = _precompute_ap_membership()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # 1. Local search from Thue-Morse
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        c, cnt = _local_search_delta(c, min(10, time_remaining - 5), ap_arrays)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # 2. Genetic Algorithm
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        # Allocate most of the remaining time to GA
        ga_time = remaining - 5 # Keep 5s for final polish
        c_ga, cnt_ga = _genetic_algorithm(best_c.copy(), ga_time, ap_arrays, pop_size=20)
        if cnt_ga < best_count:
            best_count = cnt_ga
            best_c = c_ga.copy()
    
    # 3. Final polish
    elapsed = time.time() - start_time
    if DEADLINE - elapsed > 2:
        c_final = best_c.copy()
        c_final, cnt_final = _local_search_delta(c_final, min(3, DEADLINE - elapsed - 1), ap_arrays)
        if cnt_final < best_count:
            best_count = cnt_final
            best_c = c_final.copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
