"""Optimized two-colouring of [1,256] minimizing monochromatic 4-APs."""
import numpy as np
import time
import random

N = 256

def _precompute_ap_indices():
    """Precompute all 4-term APs and which positions each position belongs to."""
    # For each position p, store list of (d, role) where role is 0,1,2,3
    # AP: a, a+d, a+2d, a+3d
    # p = a + role*d
    pos_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            # AP: a, a+d, a+2d, a+3d
            p0 = a
            p1 = a + d
            p2 = a + 2 * d
            p3 = a + 3 * d
            pos_aps[p0].append((d, 0))
            pos_aps[p1].append((d, 1))
            pos_aps[p2].append((d, 2))
            pos_aps[p3].append((d, 3))
    
    return pos_aps

def _thue_morse_fast(n):
    """Generate Thue-Morse sequence: t(i) = popcount(i) % 2."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _count_mono4_full(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N."""
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
        count += np.sum(mask)
    return int(count)

def _delta_flip(c, p, pos_aps):
    """Compute change in mono-4AP count when flipping position p.
    Returns the delta (change in count)."""
    old_val = c[p]
    new_val = 1 - old_val
    delta = 0
    
    for d, role in pos_aps[p]:
        # Get the other 3 positions
        a = p - role * d
        others = []
        for r in range(4):
            if r != role:
                others.append(c[a + r * d])
        
        was_mono = all(o == old_val for o in others)
        now_mono = all(o == new_val for o in others)
        
        if was_mono and not now_mono:
            delta -= 1
        elif now_mono and not was_mono:
            delta += 1
    
    return delta

def _local_search_incremental(c, time_limit, random_seed=42, pos_aps=None):
    """Local search with incremental delta computation."""
    if pos_aps is None:
        pos_aps = _precompute_ap_indices()
    
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_full(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        order = list(range(N))
        rng.shuffle(order)
        for pos in order:
            if (time.time() - start) >= time_limit:
                break
            delta = _delta_flip(best_c, pos, pos_aps)
            if delta < 0:
                best_c[pos] = 1 - best_c[pos]
                best_count += delta
                improved = True
    
    return best_c, best_count

def _simulated_annealing(c, time_limit, random_seed=42, pos_aps=None):
    """Simulated annealing with incremental delta computation."""
    if pos_aps is None:
        pos_aps = _precompute_ap_indices()
    
    rng = random.Random(random_seed)
    start = time.time()
    cur_c = c.copy()
    cur_count = _count_mono4_full(cur_c)
    best_c = cur_c.copy()
    best_count = cur_count
    
    T_start = 2.0
    T_end = 0.05
    
    # Estimate steps
    n_steps_target = 80000
    
    for step in range(n_steps_target):
        if (time.time() - start) >= time_limit:
            break
        progress = step / n_steps_target
        T = T_start * (T_end / T_start) ** progress
        
        pos = rng.randint(0, N - 1)
        delta = _delta_flip(cur_c, pos, pos_aps)
        
        if delta <= 0:
            cur_c[pos] = 1 - cur_c[pos]
            cur_count += delta
            if cur_count < best_count:
                best_count = cur_count
                best_c = cur_c.copy()
        else:
            if rng.random() < np.exp(-delta / T):
                cur_c[pos] = 1 - cur_c[pos]
                cur_count += delta
                if cur_count < best_count:
                    best_count = cur_count
                    best_c = cur_c.copy()
    
    return best_c, best_count

def _genetic_algorithm(starts, counts, time_limit, random_seed=42, pos_aps=None):
    """Genetic algorithm with crossover and mutation."""
    if pos_aps is None:
        pos_aps = _precompute_ap_indices()
    
    rng = random.Random(random_seed)
    start_time = time.time()
    
    # Sort by count
    paired = sorted(zip(counts, starts), key=lambda x: x[0])
    population = [c.copy() for _, c in paired[:32]]
    pop_counts = [_count_mono4_full(c) for c in population]
    
    best_c = population[0].copy()
    best_count = pop_counts[0]
    
    pop_size = len(population)
    elite_count = max(4, pop_size // 4)
    mutation_rate = 0.05
    crossover_rate = 0.7
    
    generation = 0
    while (time.time() - start_time) < time_limit:
        # Select parents (tournament selection)
        def tournament_select(k=3):
            indices = rng.sample(range(pop_size), k)
            best_idx = min(indices, key=lambda i: pop_counts[i])
            return population[best_idx].copy(), pop_counts[best_idx]
        
        new_population = []
        
        # Keep elites
        elite_indices = sorted(range(pop_size), key=lambda i: pop_counts[i])[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            if (time.time() - start_time) >= time_limit:
                break
            
            if rng.random() < crossover_rate:
                parent1, _ = tournament_select()
                parent2, _ = tournament_select()
                # Uniform crossover
                mask = (rng.random(N) < 0.5).astype(np.int8)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
            else:
                parent, _ = tournament_select()
                child = parent.copy()
            
            # Mutation
            for i in range(N):
                if rng.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            child_count = _count_mono4_full(child)
            new_population.append(child)
        
        population = new_population[:pop_size]
        pop_counts = [_count_mono4_full(c) for c in population]
        
        # Update best
        min_idx = np.argmin(pop_counts)
        if pop_counts[min_idx] < best_count:
            best_count = pop_counts[min_idx]
            best_c = population[min_idx].copy()
        
        generation += 1
    
    return best_c, best_count

def _two_flip_search(c, time_limit, random_seed=42, pos_aps=None):
    """Try flipping pairs of positions simultaneously."""
    if pos_aps is None:
        pos_aps = _precompute_ap_indices()
    
    rng = random.Random(random_seed)
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_full(best_c)
    
    improved = True
    attempts = 0
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try random pairs
        for _ in range(3000):
            if (time.time() - start) >= time_limit:
                break
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i == j:
                continue
            if best_c[i] == best_c[j]:
                continue
            
            old_i = best_c[i]
            old_j = best_c[j]
            best_c[i] = 1 - old_i
            best_c[j] = 1 - old_j
            new_count = _count_mono4_full(best_c)
            if new_count < best_count:
                best_count = new_count
                improved = True
            else:
                best_c[i] = old_i
                best_c[j] = old_j
            attempts += 1
    
    return best_c, best_count

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    rng_global = random.Random(12345)
    
    # Precompute AP indices
    pos_aps = _precompute_ap_indices()
    
    # Generate diverse starting points
    starts = []
    
    # Thue-Morse and variants
    tm = _thue_morse_fast(N)
    starts.append(tm.copy())
    for offset in range(1, 16):
        starts.append(np.roll(tm, offset).copy())
    
    # Random with bias toward 50/50
    for _ in range(15):
        r = np.random.RandomState(rng_global.randint(0, 2**31))
        starts.append(r.randint(0, 2, N).astype(np.int8))
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 6, 8]:
        pat = np.zeros(N, dtype=np.int8)
        for i in range(N):
            pat[i] = (i // period) % 2
        starts.append(pat.copy())
    
    # More structured patterns
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        for num_blocks in [2, 3, 4]:
            pat = np.zeros(N, dtype=np.int8)
            for i in range(N):
                block_idx = i // block_size
                pat[i] = (block_idx % num_blocks) % 2
            starts.append(pat.copy())
    
    best_c = None
    best_count = float('inf')
    
    # Evaluate all starts
    start_counts = []
    for s in starts:
        cnt = _count_mono4_full(s)
        start_counts.append(cnt)
        if cnt < best_count:
            best_count = cnt
            best_c = s.copy()
    
    # Phase 1: Local search on best start
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 20), 42, pos_aps)
    
    # Phase 2: Genetic algorithm on diverse starts
    remaining = deadline - time.time()
    if remaining > 30:
        ga_c, ga_count = _genetic_algorithm(starts, start_counts, min(remaining - 30, 45), 777, pos_aps)
        if ga_count < best_count:
            best_count = ga_count
            best_c = ga_c
    
    # Phase 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 20:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 20, 35), 999, pos_aps)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Phase 4: Local search again after SA
    remaining = deadline - time.time()
    if remaining > 10:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 10, 20), 111, pos_aps)
    
    # Phase 5: Two-flip search
    remaining = deadline - time.time()
    if remaining > 15:
        tf_c, tf_count = _two_flip_search(best_c, min(remaining - 15, 25), 222, pos_aps)
        if tf_count < best_count:
            best_count = tf_count
            best_c = tf_c
    
    # Phase 6: Final local search
    remaining = deadline - time.time()
    if remaining > 5:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 5, 15), 333, pos_aps)
    
    # Phase 7: One more SA pass if time
    remaining = deadline - time.time()
    if remaining > 10:
        sa_c, sa_count = _simulated_annealing(best_c, min(remaining - 10, 15), 444, pos_aps)
        if sa_count < best_count:
            best_count = sa_count
            best_c = sa_c
    
    # Final local search
    remaining = deadline - time.time()
    if remaining > 3:
        best_c, best_count = _local_search_incremental(best_c, min(remaining - 3, 10), 555, pos_aps)
    
    # Convert to list of ints
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
