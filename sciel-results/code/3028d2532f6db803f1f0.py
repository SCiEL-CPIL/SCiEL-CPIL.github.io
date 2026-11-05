# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[j]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def _makespan_fast(perm, pt, comp_cache=None):
    """Compute makespan, optionally using a pre-allocated cache."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    if comp_cache is None:
        comp = np.zeros(m, dtype=float)
    else:
        comp = comp_cache
        comp[:] = 0.0
    for j in range(n):
        job = perm[j]
        pt_row = pt[j]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def _delta_insert(perm, pt, job, pos, comp_cache):
    """
    Compute the makespan of inserting 'job' at position 'pos' in 'perm'
    using a delta approach. Returns the new makespan.
    """
    n = len(perm)
    m = pt.shape[1]
    if comp_cache is None:
        comp = np.zeros(m, dtype=float)
    else:
        comp = comp_cache
        comp[:] = 0.0
    
    # Build completion times for perm without the job
    # We need to simulate the flow shop for the sequence: perm[0:pos], job, perm[pos:]
    # But we can reuse prefix computations if we had them. For simplicity, 
    # we compute the full makespan but with a slight optimization: 
    # Actually, for delta insertion, we need to compute the completion times 
    # up to position pos-1, then insert the job, then continue.
    
    # Let's just compute the full makespan of the new permutation for correctness,
    # but we'll do it efficiently.
    # Actually, let's implement a proper delta:
    # 1. Compute completion times for perm[0:pos]
    # 2. Insert job, compute its completion times
    # 3. Continue with perm[pos:]
    
    # For now, use the straightforward method but it's O(n*m)
    # A true delta would be O(m) for the insertion point, but requires prefix/suffix tables.
    # Given the time constraints, let's use a hybrid: compute prefix completion times once.
    
    # Actually, let's just call _makespan_fast on the new permutation.
    # The overhead of list manipulation is the bottleneck, not the makespan calc.
    new_perm = list(perm)
    new_perm.insert(pos, job)
    return _makespan_fast(new_perm, pt, comp_cache)

def _neh_initial(pt):
    """NEH heuristic: build permutation greedily by inserting each job (sorted by decreasing total) at the best position."""
    n, m = pt.shape
    if n <= 1:
        return np.arange(n)
    
    total_pt = pt.sum(axis=1)
    order = np.argsort(-total_pt)
    
    perm = [int(order[0])]
    for idx in range(1, n):
        job = int(order[idx])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            trial = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(trial, pt)
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

def _insertion_ls(perm, pt, time_limit, start_time, first_improvement=True):
    """Insertion local search."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            remaining = np.delete(best_perm, i)
            for j in range(n):
                if time.time() - start_time >= time_limit:
                    break
                if i == j:
                    continue
                new_perm = np.insert(remaining, j, job)
                ms = _makespan_fast(new_perm, pt, comp_cache)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    if first_improvement:
                        i = j
                        break
    
    return best_perm, best_ms

def _or_opt_ls(perm, pt, time_limit, start_time, first_improvement=True):
    """Or-opt: move a segment of 1-3 jobs to a better position."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    comp_cache = np.zeros(pt.shape[1], dtype=float)
    
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
                segment = best_perm[i:i+seg_len]
                remaining = np.concatenate([best_perm[:i], best_perm[i+seg_len:]])
                for j in range(len(remaining) + 1):
                    if time.time() - start_time >= time_limit:
                        break
                    if j == i:
                        continue
                    new_perm = np.insert(remaining, j, segment)
                    ms = _makespan_fast(new_perm, pt, comp_cache)
                    if ms < best_ms:
                        best_ms = ms
                        best_perm = new_perm
                        improved = True
                        if first_improvement:
                            break
                if improved and first_improvement:
                    break
            if improved and first_improvement:
                break
    
    return best_perm, best_ms

def _simulated_annealing(perm, pt, start_time, deadline, n_particles=1, max_iters=10000):
    """
    Simulated Annealing with Or-Opt moves.
    """
    n = len(perm)
    m = pt.shape[1]
    comp_cache = np.zeros(m, dtype=float)
    
    current_perm = perm.copy()
    current_ms = _makespan_fast(current_perm, pt, comp_cache)
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    T_start = 10.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() - start_time >= deadline:
            break
        
        # Generate a random Or-Opt move
        seg_len = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        if seg_len > n:
            seg_len = 1
        
        i = np.random.randint(0, n - seg_len + 1)
        j = np.random.randint(0, n - seg_len + 1)
        if j == i:
            continue
        
        # Extract segment
        segment = current_perm[i:i+seg_len]
        remaining = np.concatenate([current_perm[:i], current_perm[i+seg_len:]])
        
        # Adjust insertion position
        if j > i:
            j = j - seg_len
        
        new_perm = np.insert(remaining, j, segment)
        new_ms = _makespan_fast(new_perm, pt, comp_cache)
        
        # Acceptance criterion
        delta = new_ms - current_ms
        if delta < 0 or np.random.rand() < np.exp(-delta / T_start):
            current_perm = new_perm
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = new_perm.copy()
        
        T_start *= cooling
    
    return best_perm, best_ms

def _vnd_search(perm, pt, start_time, deadline):
    """
    Variable Neighborhood Descent:
    1. Insertion LS
    2. Or-Opt LS
    3. Reversal LS
    """
    current_perm = perm.copy()
    current_ms = _makespan(current_perm, pt)
    
    # Phase 1: Insertion
    current_perm, current_ms = _insertion_ls(current_perm, pt, 15.0, start_time)
    
    # Phase 2: Or-Opt
    if time.time() - start_time < deadline:
        current_perm, current_ms = _or_opt_ls(current_perm, pt, 10.0, start_time)
    
    # Phase 3: Reversal
    if time.time() - start_time < deadline:
        n = len(current_perm)
        comp_cache = np.zeros(pt.shape[1], dtype=float)
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            for i in range(n):
                if time.time() - start_time >= deadline:
                    break
                for j in range(i + 2, n):
                    if time.time() - start_time >= deadline:
                        break
                    new_perm = current_perm.copy()
                    new_perm[i:j+1] = new_perm[i:j+1][::-1]
                    ms = _makespan_fast(new_perm, pt, comp_cache)
                    if ms < current_ms:
                        current_ms = ms
                        current_perm = new_perm
                        improved = True
                        break
                if improved:
                    break
    
    return current_perm, current_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 145.0  # Safety margin of 5s
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    comp_cache = np.zeros(m, dtype=float)
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # NEH
    perm_neh = _neh_initial(processing_times)
    initial_solutions.append(perm_neh)
    
    # Palmer
    perm_palmer = _palmer_initial(processing_times)
    initial_solutions.append(perm_palmer)
    
    # Simple NEH (sort by total)
    total_pt = processing_times.sum(axis=1)
    perm_simple = np.argsort(-total_pt).astype(int)
    initial_solutions.append(perm_simple)
    
    # Bottleneck
    machine_totals = processing_times.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    perm_bn = np.argsort(-processing_times[:, bottleneck_machine]).astype(int)
    initial_solutions.append(perm_bn)
    
    # First machine
    perm_first = np.argsort(-processing_times[:, 0]).astype(int)
    initial_solutions.append(perm_first)
    
    # Last machine
    perm_last = np.argsort(-processing_times[:, -1]).astype(int)
    initial_solutions.append(perm_last)
    
    # Random solutions
    for _ in range(5):
        perm_rand = rng.permutation(n)
        initial_solutions.append(perm_rand)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    
    for perm in initial_solutions:
        ms = _makespan_fast(perm, processing_times, comp_cache)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Apply VND to the best initial solution
    best_perm, best_ms = _vnd_search(best_perm, processing_times, start_time, deadline)
    
    # Random restarts with SA
    remaining_time = deadline - (time.time() - start_time)
    num_restarts = min(3, int(remaining_time // 30))
    
    for r in range(num_restarts):
        if time.time() - start_time >= deadline:
            break
        
        # Generate a random initial solution
        perm_rand = rng.permutation(n)
        
        # Apply SA
        sa_perm, sa_ms = _simulated_annealing(perm_rand, processing_times, start_time, deadline, max_iters=2000)
        
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm.copy()
        
        # Apply VND to the SA result
        vnd_perm, vnd_ms = _vnd_search(sa_perm, processing_times, start_time, deadline)
        
        if vnd_ms < best_ms:
            best_ms = vnd_ms
            best_perm = vnd_perm.copy()
    
    # Final polish with Or-Opt
    if time.time() - start_time < deadline:
        best_perm, best_ms = _or_opt_ls(best_perm, processing_times, 5.0, start_time)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
