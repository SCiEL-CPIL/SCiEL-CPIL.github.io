# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Optimized makespan computation using pure Python loops for small n."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    # Use list-based approach for better cache performance
    prev = [0.0] * n
    # Machine 0
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


def _makespan_vectorized(perm, pt):
    """Vectorized makespan for larger problems."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev_completion = np.cumsum(pt[perm, 0])
    
    for k in range(1, m):
        curr_completion = np.empty(n, dtype=np.float64)
        curr_completion[0] = prev_completion[0] + pt[perm[0], k]
        for i in range(1, n):
            curr_completion[i] = max(prev_completion[i], curr_completion[i-1]) + pt[perm[i], k]
        prev_completion = curr_completion
    
    return float(prev_completion[-1])


def _makespan(perm, pt):
    """Dispatch to the appropriate makespan function."""
    n = len(perm)
    if n <= 200:
        return _makespan_fast(perm, pt)
    else:
        return _makespan_vectorized(perm, pt)


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
    
    # Build permutation greedily
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


def _local_search_2opt(perm, pt, max_iters=500, deadline=None):
    """2-opt local search for flow shop with time limit."""
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
                new_perm[i:j+1] = best_perm[i:j+1][::-1]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
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
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
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
            # Remove job i
            job = best_perm[i]
            remaining = np.delete(best_perm, i)
            for pos in range(len(remaining) + 1):
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
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
    
    # Try insertion local search
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=100, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick local search from random start
            ls_r = _local_search_swap(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try 2-opt as well
            if time.time() < deadline:
                ls_r2 = _local_search_2opt(ls_r, pt, max_iters=50, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm = _local_search_swap(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with 2-opt
    if time.time() < deadline:
        final_perm2 = _local_search_2opt(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    return list(best_perm)
# EVOLVE-BLOCK-END
