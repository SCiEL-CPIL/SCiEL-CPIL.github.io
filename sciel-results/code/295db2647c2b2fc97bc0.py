# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    if m == 1:
        return float(pt[perm].sum())
    # For small m, use list for speed
    if m <= 8:
        comps = [0.0] * m
        for j in range(n):
            job = perm[j]
            pt_row = pt[job]
            for mi in range(m):
                prev = comps[mi - 1] if mi > 0 else 0.0
                comps[mi] = (comps[mi] if comps[mi] > prev else prev) + pt_row[mi]
        return comps[-1]
    else:
        comp = np.zeros(m, dtype=float)
        for j in range(n):
            job = perm[j]
            pt_row = pt[job]
            for mi in range(m):
                prev = comp[mi - 1] if mi > 0 else 0.0
                comp[mi] = (comp[mi] if comp[mi] > prev else prev) + pt_row[mi]
        return comp[-1]

def _makespan_fast(perm, pt, comp_cache=None):
    """Compute makespan, optionally using a pre-allocated cache."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    if m == 1:
        return float(pt[perm].sum())
    if m <= 8:
        comps = [0.0] * m
        for j in range(n):
            job = perm[j]
            pt_row = pt[job]
            for mi in range(m):
                prev = comps[mi - 1] if mi > 0 else 0.0
                comps[mi] = (comps[mi] if comps[mi] > prev else prev) + pt_row[mi]
        return comps[-1]
    else:
        if comp_cache is None:
            comp = np.zeros(m, dtype=float)
        else:
            comp = comp_cache
            comp[:] = 0.0
        for j in range(n):
            job = perm[j]
            pt_row = pt[job]
            for mi in range(m):
                prev = comp[mi - 1] if mi > 0 else 0.0
                comp[mi] = (comp[mi] if comp[mi] > prev else prev) + pt_row[mi]
        return comp[-1]

def _delta_insertion(perm, pt, job, pos):
    """
    Compute the change in makespan when inserting job at position pos in perm.
    This is an O(n*m) computation but avoids building a new permutation array.
    """
    n = len(perm)
    m = pt.shape[1]
    if m == 1:
        return float(pt[job, 0])
    
    # We need to compute the makespan of perm with job inserted at pos
    # and compare with makespan of perm.
    # Actually, for simplicity and correctness, let's just compute both.
    # A true delta computation is complex. Let's just compute the new makespan.
    # To speed it up, we can compute the completion times of the original perm
    # and then "replay" from the insertion point.
    
    # For now, let's just use the full makespan computation on the modified perm.
    # This is O(n*m) which is the same as _makespan_fast.
    # The benefit is avoiding the np.insert array copy.
    
    if m <= 8:
        comps = [0.0] * m
        for k in range(n + 1):
            if k < pos:
                job_k = perm[k]
            elif k == pos:
                job_k = job
            else:
                job_k = perm[k - 1]
            pt_row = pt[job_k]
            for mi in range(m):
                prev = comps[mi - 1] if mi > 0 else 0.0
                comps[mi] = (comps[mi] if comps[mi] > prev else prev) + pt_row[mi]
        return comps[-1]
    else:
        comp = np.zeros(m, dtype=float)
        for k in range(n + 1):
            if k < pos:
                job_k = perm[k]
            elif k == pos:
                job_k = job
            else:
                job_k = perm[k - 1]
            pt_row = pt[job_k]
            for mi in range(m):
                prev = comp[mi - 1] if mi > 0 else 0.0
                comp[mi] = (comp[mi] if comp[mi] > prev else prev) + pt_row[mi]
        return comp[-1]

def _neh_initial(pt):
    """NEH heuristic: build permutation greedily by inserting each job (sorted by decreasing total) at the best position."""
    n, m = pt.shape
    if n <= 1:
        return np.arange(n)
    
    total_pt = pt.sum(axis=1)
    order = np.argsort(-total_pt)
    
    perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            ms = _delta_insertion(perm, pt, job, pos)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=int)

def _palmer_initial(pt):
    """Palmer's heuristic."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _random_initial(pt, rng):
    """Random initial permutation."""
    n, _ = pt.shape
    return rng.permutation(n)

def _insertion_ls(perm, pt, time_limit, start_time):
    """Insertion local search with fast makespan evaluation."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    m = pt.shape[1]
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            # Try inserting job at every other position
            for j in range(n):
                if i == j:
                    continue
                # Adjust position: if j > i, the job shifts left by one when removed
                # So we're trying to insert at position j in the sequence without job i
                # which corresponds to position j in the new array
                # Actually, let's just build the new permutation and check
                # To avoid array manipulation, use _delta_insertion
                # But _delta_insertion assumes job is not in perm.
                # Let's use the full makespan for correctness in LS.
                
                # Create new permutation by removing job at i and inserting at j
                # If j > i, removing shifts left, so insert at j-1 in the remaining
                # If j <= i, insert at j in the remaining
                # Let's just do it explicitly for clarity
                remaining = list(best_perm[:i]) + list(best_perm[i+1:])
                insert_pos = j if j <= i else j - 1
                if insert_pos >= len(remaining):
                    insert_pos = len(remaining)
                
                # Compute makespan of remaining with job inserted at insert_pos
                ms = _delta_insertion(np.array(remaining, dtype=int), pt, job, insert_pos)
                
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = np.array(remaining[:insert_pos] + [job] + remaining[insert_pos:], dtype=int)
                    improved = True
                    break
    
    return best_perm, best_ms

def _or_opt_ls(perm, pt, time_limit, start_time):
    """Or-opt: move a segment of 1-3 jobs to a better position."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    m = pt.shape[1]
    
    improved = True
    seg_lens = [1, 2, 3]
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        for seg_len in seg_lens:
            if time.time() - start_time >= time_limit:
                break
            for i in range(n - seg_len + 1):
                if time.time() - start_time >= time_limit:
                    break
                # Extract segment
                segment = best_perm[i:i+seg_len].copy()
                # Remove segment
                remaining = np.concatenate([best_perm[:i], best_perm[i+seg_len:]])
                for j in range(len(remaining) + 1):
                    if time.time() - start_time >= time_limit:
                        break
                    if j == i or (j > i and j <= i + seg_len):
                        continue  # Skip inserting back in original position
                    # Build new permutation
                    new_perm = np.concatenate([remaining[:j], segment, remaining[j:]])
                    ms = _makespan_fast(new_perm, pt)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return best_perm, best_ms

def _simulated_annealing(perm, pt, start_time, time_budget):
    """Simulated annealing with insertion moves."""
    n, m = pt.shape
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    current_perm = perm.copy()
    current_ms = _makespan(current_perm, pt)
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    T0 = current_ms * 0.1 if current_ms > 0 else 10.0
    T_end = T0 * 1e-6
    total_time = time_budget
    elapsed0 = time.time() - start_time
    remaining = total_time - elapsed0
    if remaining <= 0:
        return best_perm, best_ms
    
    # Estimate iterations
    # Each iteration: one makespan computation O(n*m) + overhead
    # Let's just run until time runs out
    T = T0
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_budget:
        # Generate neighbor: swap two random positions
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan_fast(new_perm, pt)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        
        # Cool down
        T = max(T_end, T * 0.999)
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # NEH (greedy insertion)
    perm_neh = _neh_initial(processing_times)
    initial_solutions.append(perm_neh)
    
    # Palmer
    perm_palmer = _palmer_initial(processing_times)
    initial_solutions.append(perm_palmer)
    
    # Random solutions
    for _ in range(6):
        perm_rand = _random_initial(processing_times, rng)
        initial_solutions.append(perm_rand)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    
    for perm in initial_solutions:
        ms = _makespan_fast(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Local search phases
    # Phase 1: Insertion LS
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 5.0:
        alloc = min(remaining * 0.35, 40.0)
        best_perm, best_ms = _insertion_ls(best_perm, processing_times, alloc, start_time)
    
    # Phase 2: Or-opt LS
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 5.0:
        alloc = min(remaining * 0.25, 30.0)
        best_perm, best_ms = _or_opt_ls(best_perm, processing_times, alloc, start_time)
    
    # Phase 3: Simulated Annealing
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 10.0:
        alloc = min(remaining * 0.3, 50.0)
        sa_perm, sa_ms = _simulated_annealing(best_perm, processing_times, start_time, alloc + (time.time() - start_time))
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
    
    # Phase 4: Final Insertion LS
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 5.0:
        alloc = min(remaining * 0.5, 30.0)
        best_perm, best_ms = _insertion_ls(best_perm, processing_times, alloc, start_time)
    
    # Phase 5: Final Or-opt polish
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 3.0:
        alloc = min(remaining * 0.8, 20.0)
        best_perm, best_ms = _or_opt_ls(best_perm, processing_times, alloc, start_time)
    
    # Random restarts if time permits
    remaining = effective_deadline - (time.time() - start_time)
    if remaining > 20.0:
        num_restarts = min(3, int(remaining // 20))
        for _ in range(num_restarts):
            if time.time() - start_time >= effective_deadline:
                break
            perm_rand = _random_initial(processing_times, rng)
            # Quick insertion LS
            alloc = min(10.0, effective_deadline - (time.time() - start_time))
            if alloc > 2.0:
                perm_rand, ms_rand = _insertion_ls(perm_rand, processing_times, alloc, start_time)
                if ms_rand < best_ms:
                    best_ms = ms_rand
                    best_perm = perm_rand.copy()
            # Or-opt
            remaining = effective_deadline - (time.time() - start_time)
            if remaining > 5.0:
                alloc = min(8.0, remaining)
                if alloc > 2.0:
                    perm_rand, ms_rand = _or_opt_ls(perm_rand, processing_times, alloc, start_time)
                    if ms_rand < best_ms:
                        best_ms = ms_rand
                        best_perm = perm_rand.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
