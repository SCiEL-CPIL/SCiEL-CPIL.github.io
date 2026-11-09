"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]

def _makespan_vec(perm, pt):
    """Vectorized makespan for larger instances."""
    if len(perm) == 0:
        return 0.0
    pt_perm = pt[perm]
    m = pt_perm.shape[1]
    # Compute completion times iteratively but with numpy for each job
    c = np.zeros(m)
    for job_idx in range(len(perm)):
        row = pt_perm[job_idx]
        # c[k] = max(c[k], c[k-1]) + row[k], with c[-1]=0
        prev = 0.0
        for k in range(m):
            val = c[k]
            if val < prev:
                val = prev
            val += row[k]
            c[k] = val
            prev = val
    return float(c[-1])

def _neh(processing_times):
    """NEH heuristic to generate initial solution."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals).tolist()
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial_perm = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm

def _insertion_local_search(perm, processing_times, max_passes=20, time_limit=None, start_time=None):
    """Local search using insertion and swap moves with early stopping."""
    n = len(perm)
    current_ms = _makespan_fast(perm, processing_times)
    
    if start_time is None:
        start_time = time.time()
    
    for _ in range(max_passes):
        if time_limit is not None and time.time() > time_limit:
            break
        improved = False
        
        # Try insertion moves
        for i in range(n):
            if time_limit is not None and time.time() > time_limit:
                break
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                new_perm = perm.copy()
                new_perm.pop(i)
                adj_j = j - 1 if i < j else j
                if adj_j < 0 or adj_j >= len(new_perm):
                    continue
                new_perm.insert(adj_j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        
        # Try swap moves if no insertion improvement
        if not improved:
            for i in range(n - 1):
                if time_limit is not None and time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    new_perm = perm.copy()
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_ms = _makespan_fast(new_perm, processing_times)
                    if new_ms < current_ms:
                        perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return perm, current_ms

def _sa_insertion(perm, pt, T_init, T_min, alpha, deadline, best_perm, best_ms):
    """Simulated Annealing with insertion and swap moves."""
    n = len(perm)
    current_perm = perm.copy()
    current_ms = _makespan_fast(current_perm, pt)
    
    if current_ms < best_ms:
        best_ms = current_ms
        best_perm = current_perm.copy()
    
    T = T_init
    stall_count = 0
    max_stall = 300
    
    while time.time() < deadline and T > T_min:
        # Choose move type: insertion (70%) or swap (30%)
        move_type = random.random()
        
        if move_type < 0.7:
            # Insertion move
            i = random.randint(0, n - 1)
            if n <= 1:
                break
            j = random.randint(0, n - 1)
            if j == i:
                continue
            
            new_perm = current_perm.copy()
            new_perm.pop(i)
            adj_j = j - 1 if i < j else j
            if adj_j < 0 or adj_j >= len(new_perm):
                continue
            new_perm.insert(adj_j, current_perm[i])
        else:
            # Swap move
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        new_ms = _makespan_fast(new_perm, pt)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            stall_count = 0
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        else:
            stall_count += 1
        
        T *= alpha
        
        if stall_count > max_stall:
            break
    
    return current_perm, current_ms, best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Generate multiple diverse initial solutions using NEH with different orderings
    candidates = []
    
    # Standard NEH (descending total processing time)
    perm1 = _neh(processing_times)
    candidates.append(perm1)
    
    # NEH with ascending order (reverse)
    totals = processing_times.sum(axis=1)
    order_asc = np.argsort(totals).tolist()
    perm2 = []
    for job in order_asc:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm2) + 1):
            trial_perm = perm2[:k] + [job] + perm2[k:]
            ms = _makespan_fast(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm2.insert(best_pos, job)
    candidates.append(perm2)
    
    # NEH with random tie-breaking
    for _ in range(2):
        random_order = list(range(n))
        random.shuffle(random_order)
        perm_rand = []
        for job in random_order:
            best_pos = 0
            best_ms = float('inf')
            for k in range(len(perm_rand) + 1):
                trial_perm = perm_rand[:k] + [job] + perm_rand[k:]
                ms = _makespan_fast(trial_perm, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = k
            perm_rand.insert(best_pos, job)
        candidates.append(perm_rand)
    
    # Random permutations
    for _ in range(2):
        random_perm = list(range(n))
        random.shuffle(random_perm)
        candidates.append(random_perm)
    
    # Polish each candidate and keep the best
    best_perm = None
    best_ms = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 40.0:
            break
        polished, ms = _insertion_local_search(cand, processing_times, max_passes=8, time_limit=deadline - 40.0, start_time=start_time)
        if ms < best_ms:
            best_ms = ms
            best_perm = polished
    
    # 2. Multi-restart SA with insertion and swap moves
    random.seed(42)
    pt = processing_times
    
    T_init = best_ms * 0.15
    T_min = best_ms * 0.0005
    alpha = 0.997
    
    n_restarts = 10
    remaining_time = deadline - time.time() - 20.0
    if remaining_time <= 0:
        return list(best_perm)
    time_per_restart = remaining_time / max(1, n_restarts)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # For restart 0, use the best found so far
        if restart == 0:
            current_perm = best_perm.copy()
        else:
            # Perturb the best solution with random moves
            current_perm = best_perm.copy()
            n_perturb = min(n, random.randint(3, max(3, n // 2)))
            for _ in range(n_perturb):
                move_type = random.random()
                if move_type < 0.7:
                    # Insertion
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    job = current_perm[i]
                    current_perm.pop(i)
                    adj_j = j - 1 if i < j else j
                    if 0 <= adj_j < len(current_perm):
                        current_perm.insert(adj_j, job)
                else:
                    # Swap
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Run SA for this restart
        restart_deadline = min(deadline - 20.0, time.time() + time_per_restart)
        current_perm, current_ms, best_perm, best_ms = _sa_insertion(
            current_perm, pt, T_init, T_min, alpha, restart_deadline, best_perm, best_ms
        )
        
        # Polish the best found in this restart
        if time.time() < deadline - 15.0:
            polished_perm, polished_ms = _insertion_local_search(best_perm.copy(), pt, max_passes=12, time_limit=deadline - 15.0, start_time=start_time)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final intensive polish
    if time.time() < deadline - 5.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm.copy(), pt, max_passes=20, time_limit=deadline - 5.0, start_time=start_time)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
