"""Low discrepancy on all arithmetic progressions: optimized local search."""
import time
import random
import numpy as np

N = 256
TIME_LIMIT = 85.0  # safety margin before 90s kill

def _build_ap_membership():
    """
    Build for each position the list of AP indices it belongs to.
    Also build the list of all APs as index arrays.
    Returns: ap_index_lists (list of index arrays), pos_to_aps (list of lists of AP indices)
    """
    ap_index_lists = []
    # We'll also need a mapping from (start, step, length) to AP index
    # For efficiency, let's enumerate all APs
    
    for step in range(1, N):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 1:
                continue
            # All lengths from 1 to max_len
            for length in range(1, max_len + 1):
                indices = [start - 1 + k * step for k in range(length)]
                ap_index_lists.append(indices)
    
    # Build position to AP mapping
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_index_lists):
        for i in indices:
            pos_to_aps[i].append(ap_idx)
    
    return ap_index_lists, pos_to_aps

def _compute_max_abs_sum_fast(signs, ap_index_lists):
    """Compute max absolute sum over all APs using numpy."""
    max_abs = 0
    for indices in ap_index_lists:
        s = 0
        for i in indices:
            s += signs[i]
        a = abs(s)
        if a > max_abs:
            max_abs = a
    return max_abs

def _compute_all_ap_sums(signs, ap_index_lists):
    """Compute sums for all APs, return array."""
    sums = np.zeros(len(ap_index_lists), dtype=np.int32)
    for ap_idx, indices in enumerate(ap_index_lists):
        s = 0
        for i in indices:
            s += signs[i]
        sums[ap_idx] = s
    return sums

def _sa_search(signs, ap_index_lists, pos_to_aps, time_limit, rng, start_temp=5.0, decay=0.999):
    """
    Simulated annealing to minimize max absolute AP sum.
    """
    signs = np.array(signs, dtype=np.int8)
    sums = _compute_all_ap_sums(signs, ap_index_lists)
    current_max = int(np.max(np.abs(sums)))
    best_max = current_max
    best_signs = signs.copy()
    
    deadline = time.time() + time_limit
    T = start_temp
    steps = 0
    
    while time.time() < deadline - 1.0:
        # Pick a random position to flip
        pos = rng.randint(0, N - 1)
        old_sign = int(signs[pos])
        new_sign = -old_sign
        
        # Compute delta for all APs containing pos
        affected_aps = pos_to_aps[pos]
        
        # For each affected AP, the sum changes by 2*new_sign (since we flip)
        # delta for AP j: sums[j] -> sums[j] - old_sign + new_sign = sums[j] - 2*old_sign
        # Since new_sign = -old_sign, this is sums[j] + 2*new_sign
        
        # Compute new max efficiently
        # Store old sums for affected APs
        old_sums_affected = sums[affected_aps].copy()
        
        # Apply flip
        sums[affected_aps] += 2 * new_sign
        
        new_max = int(np.max(np.abs(sums)))
        
        if new_max < current_max:
            # Accept
            current_max = new_max
            signs[pos] = new_sign
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            delta = new_max - current_max
            if delta == 0:
                # Plateau, accept to allow exploration
                signs[pos] = new_sign
                current_max = new_max
            else:
                # Accept with probability exp(-delta / T)
                if T > 0.01:
                    accept_prob = np.exp(-delta / T)
                    if rng.random() < accept_prob:
                        signs[pos] = new_sign
                        current_max = new_max
                    else:
                        # Revert
                        sums[affected_aps] = old_sums_affected
                else:
                    # Revert
                    sums[affected_aps] = old_sums_affected
        
        T *= decay
        steps += 1
        
        if steps % 5000 == 0 and time.time() > deadline - 5.0:
            break
    
    return best_signs.tolist(), best_max

def _ga_search(pop_size, generations, time_limit, rng):
    """
    Simple genetic algorithm.
    """
    start = time.time()
    deadline = start + time_limit
    
    # Build AP data once
    ap_index_lists, pos_to_aps = _build_ap_membership()
    
    # Initial population
    pop = []
    for _ in range(pop_size):
        if rng.random() < 0.3:
            # Thue-Morse
            s = []
            for i in range(N):
                c = bin(i).count('1')
                s.append(1 if c % 2 == 0 else -1)
        elif rng.random() < 0.5:
            # Alternating
            phase = rng.randint(0, 1)
            s = [1 if (i + phase) % 2 == 0 else -1 for i in range(N)]
        else:
            # Random
            s = [rng.choice((-1, 1)) for _ in range(N)]
        pop.append(s)
    
    best_signs = None
    best_val = float('inf')
    
    for gen in range(generations):
        if time.time() > deadline - 5.0:
            break
        
        # Evaluate
        scored = []
        for s in pop:
            v = _compute_max_abs_sum_fast(s, ap_index_lists)
            scored.append((v, s))
        
        scored.sort(key=lambda x: x[0])
        
        if scored[0][0] < best_val:
            best_val = scored[0][0]
            best_signs = scored[0][1][:]
        
        # Selection: keep top half
        survivors = [s for _, s in scored[:pop_size // 2]]
        
        # Crossover and mutation to fill population
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            if len(new_pop) >= 2:
                p1, p2 = rng.sample(new_pop, 2)
                # Uniform crossover
                child = [p1[i] if rng.random() < 0.5 else p2[i] for i in range(N)]
            else:
                child = new_pop[0][:]
            # Mutation
            for i in range(N):
                if rng.random() < 0.05:
                    child[i] = -child[i]
            new_pop.append(child)
        
        pop = new_pop
    
    if best_signs is None:
        best_signs = [1] * N
        best_val = _compute_max_abs_sum_fast(best_signs, ap_index_lists)
    
    return best_signs, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    rng = random.Random(12345)
    
    # Build AP data
    ap_index_lists, pos_to_aps = _build_ap_membership()
    build_time = time.time() - start_time
    
    best_signs = None
    best_val = float('inf')
    
    # Phase 1: Try diverse initial solutions and pick the best
    initial_candidates = []
    # Thue-Morse
    tm = []
    for i in range(N):
        c = bin(i).count('1')
        tm.append(1 if c % 2 == 0 else -1)
    initial_candidates.append(tm)
    
    # Alternating
    for phase in range(2):
        initial_candidates.append([1 if (i + phase) % 2 == 0 else -1 for i in range(N)])
    
    # Random
    for _ in range(10):
        initial_candidates.append([rng.choice((-1, 1)) for _ in range(N)])
    
    for cand in initial_candidates:
        v = _compute_max_abs_sum_fast(cand, ap_index_lists)
        if v < best_val:
            best_val = v
            best_signs = cand[:]
    
    # Phase 2: Simulated annealing from best initial
    remaining = deadline - time.time()
    if remaining > 10.0:
        sa_time = min(remaining * 0.6, 40.0)
        sa_signs, sa_val = _sa_search(best_signs, ap_index_lists, pos_to_aps, sa_time, rng)
        if sa_val < best_val:
            best_val = sa_val
            best_signs = sa_signs
    
    # Phase 3: Another SA run from a different starting point
    remaining = deadline - time.time()
    if remaining > 10.0 and best_val > 5:
        # Start from a random initialization
        rand_init = [rng.choice((-1, 1)) for _ in range(N)]
        sa_time = min(remaining * 0.5, 25.0)
        sa_signs, sa_val = _sa_search(rand_init, ap_index_lists, pos_to_aps, sa_time, rng, start_temp=8.0)
        if sa_val < best_val:
            best_val = sa_val
            best_signs = sa_signs
    
    # Phase 4: Final local search (greedy)
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Greedy: keep flipping any position that reduces max
        signs = best_signs[:]
        sums = _compute_all_ap_sums(signs, ap_index_lists)
        current_max = int(np.max(np.abs(sums)))
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for pos in range(N):
                old_sign = signs[pos]
                new_sign = -old_sign
                affected = pos_to_aps[pos]
                old_sums_affected = sums[affected].copy()
                sums[affected] += 2 * new_sign
                new_max = int(np.max(np.abs(sums)))
                if new_max < current_max:
                    signs[pos] = new_sign
                    current_max = new_max
                    improved = True
                    if current_max < best_val:
                        best_val = current_max
                        best_signs = signs[:]
                else:
                    sums[affected] = old_sums_affected
    
    return [int(s) for s in best_signs]
