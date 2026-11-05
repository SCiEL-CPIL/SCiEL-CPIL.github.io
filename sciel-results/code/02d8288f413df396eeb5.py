"""Flow-shop permutation via fast heuristics, insertion local search, and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, pt_rows, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use Python lists for speed
    prev = [0.0] * n_machines
    for j in perm:
        row = pt_rows[j]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _makespan_delta_insert(pt, pt_rows, perm, job, pos):
    """
    Compute the new makespan when inserting 'job' at position 'pos' in 'perm'.
    Uses pre-extracted rows for speed.
    """
    n_machines = pt.shape[1]
    
    # Build the new permutation
    new_perm = perm[:pos] + [job] + perm[pos:]
    
    prev = [0.0] * n_machines
    for j in new_perm:
        row = pt_rows[j]
        curr0 = prev[0] + row[0]
        prev[0] = curr0
        for m in range(1, n_machines):
            c = curr0
            p = prev[m]
            if p > c:
                c = p
            c += row[m]
            prev[m] = c
            curr0 = c
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic: sort by total processing time descending, insert each job at best position."""
    n_jobs, n_machines = pt.shape
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [int(order[0])]
    for i in range(1, n_jobs):
        job = int(order[i])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, pt_rows, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer_initial(pt):
    """Palmer's heuristic: sort by priority key."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    """LPT: sort by total processing time descending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    """SPT: sort by total processing time ascending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _random_initial(n_jobs, rng):
    """Random permutation."""
    return list(rng.permutation(n_jobs))


def _insertion_local_search(pt, pt_rows, perm, deadline, max_iter=100000):
    """
    Local search using insertion neighborhood with first-improvement.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    eval_count = 0
    
    while eval_count < max_iter:
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n_jobs):
            if time.time() > deadline or eval_count >= max_iter:
                break
            
            job = current_perm[i]
            temp_perm = current_perm[:i] + current_perm[i+1:]
            
            for j in range(len(temp_perm) + 1):
                if j == i:
                    continue
                
                new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                new_ms = _makespan(pt, pt_rows, new_perm)
                eval_count += 1
                
                if new_ms < current_ms:
                    current_ms = new_ms
                    current_perm = new_perm
                    improved = True
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                    break
        
        if not improved:
            break
    
    return best_perm, best_ms


def _simulated_annealing(pt, pt_rows, perm, deadline, max_iter=20000):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random job to move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        temp_perm = current_perm[:i] + current_perm[i+1:]
        
        # Pick random insertion position
        j = random.randint(0, len(temp_perm))
        if j == i:
            continue  # No change
        
        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
        new_ms = _makespan(pt, pt_rows, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard: 145s to leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Pre-extract job rows as Python lists for faster access in makespan computation
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Random permutations
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, pt_rows, best_perm)
    
    # Run insertion local search on each candidate
    for cand in candidates:
        if time.time() > deadline - 20:  # Leave 20s for SA
            break
        
        perm, ms = _insertion_local_search(pt, pt_rows, cand, deadline - 20, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Run simulated annealing from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, pt_rows, best_perm, deadline - 5, max_iter=50000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:3]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, pt_rows, cand, deadline - 2, max_iter=10000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    return best_perm

# EVOLVE-BLOCK-END
