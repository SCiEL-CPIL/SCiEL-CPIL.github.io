# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Compute makespan for a given permutation."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev = [0.0] * n
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    for k in range(1, m):
        curr = [0.0] * n
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _completion_matrix(perm, pt):
    """Compute full completion time matrix C[job_pos][machine]."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return np.zeros((n, 0), dtype=np.float64)
    
    C = np.zeros((n, m), dtype=np.float64)
    C[0, 0] = pt[perm[0], 0]
    for i in range(1, n):
        C[i, 0] = C[i-1, 0] + pt[perm[i], 0]
    
    for k in range(1, m):
        C[0, k] = C[0, k-1] + pt[perm[0], k]
        for i in range(1, n):
            C[i, k] = max(C[i-1, k], C[i, k-1]) + pt[perm[i], k]
    
    return C


def _makespan_from_C(C):
    """Extract makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return float(C[-1, -1])


def _neh_heuristic(pt, sort_key='sum'):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    if sort_key == 'sum':
        totals = pt.sum(axis=1)
    elif sort_key == 'mean':
        totals = pt.mean(axis=1)
    elif sort_key == 'min':
        totals = pt.min(axis=1)
    elif sort_key == 'max':
        totals = pt.max(axis=1)
    else:
        totals = pt.sum(axis=1)
    
    order = np.argsort(-totals)
    
    perm = []
    
    for job in order:
        if not perm:
            perm = [job]
            continue
        
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)


def _local_search_2opt(perm, pt, max_iters=300, deadline=None):
    """2-opt local search for flow shop with time limit."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    C = _completion_matrix(best_perm, pt)
    best_ms = _makespan_from_C(C)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                # Try reversal
                new_perm = best_perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    C = _completion_matrix(best_perm, pt)
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_swap(perm, pt, max_iters=200, deadline=None):
    """Swap-based local search for flow shop."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            if deadline is not None and time.time() > deadline:
                return best_perm
            for j in range(i + 1, n):
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_insert(perm, pt, max_iters=100, deadline=None):
    """Insertion-based local search: move one job to a different position."""
    n = len(perm)
    if n <= 2:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_perm
            job = best_perm[i]
            # Build list without job at position i
            remaining = list(best_perm[:i]) + list(best_perm[i+1:])
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue  # Skip same position
                new_perm = np.array(remaining[:pos] + [job] + remaining[pos:], dtype=np.int64)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
    
    return best_perm


def _simulated_annealing(perm, pt, deadline, n_restarts=5):
    """Simulated annealing for flow shop to escape local optima."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        current_perm = best_perm.copy() if restart == 0 else np.random.permutation(n).astype(np.int64)
        current_ms = _makespan_fast(current_perm, pt)
        
        # SA parameters
        T = current_ms * 0.1  # initial temperature
        cooling_rate = 0.95
        min_T = 1e-6
        max_iterations = 500
        
        for it in range(max_iterations):
            if time.time() >= deadline:
                break
            if T < min_T:
                break
            
            # Generate neighbor by random swap
            i, j = np.random.choice(n, 2, replace=False)
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_ms = _makespan_fast(new_perm, pt)
            
            delta = new_ms - current_ms
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_perm = current_perm.copy()
                    best_ms = current_ms
            
            T *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with multiple NEH variants
    best_perm = None
    best_ms = float('inf')
    
    for sort_key in ['sum', 'mean', 'min', 'max']:
        if time.time() >= deadline:
            break
        perm = _neh_heuristic(pt, sort_key)
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_perm = perm
            best_ms = ms
    
    if best_perm is None:
        best_perm = _neh_heuristic(pt, 'sum')
        best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with 2-opt
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=300, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try insertion local search (often more effective for flow shop)
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=100, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=150, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Simulated annealing to escape local optima
    if time.time() < deadline:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, n_restarts=3)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 6
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start using insertion (more effective)
            ls_r = _local_search_insert(random_perm, pt, max_iters=40, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try swap local search as well
            if time.time() < deadline and ls_ms_r < best_ms + 1e-9:
                ls_r2 = _local_search_swap(ls_r, pt, max_iters=50, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish with insertion local search (tighter tolerance, more iterations)
    if time.time() < deadline:
        final_perm = _local_search_insert(best_perm, pt, max_iters=150, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm2 = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    # Final polish with 2-opt
    if time.time() < deadline:
        final_perm3 = _local_search_2opt(best_perm, pt, max_iters=80, deadline=deadline)
        final_ms3 = _makespan_fast(final_perm3, pt)
        if final_ms3 < best_ms:
            best_perm = final_perm3
            best_ms = final_ms3
    
    return list(best_perm)
# EVOLVE-BLOCK-END
