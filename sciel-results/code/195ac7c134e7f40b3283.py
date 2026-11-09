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


def _insertion_ls(perm, pt, max_iters=50, deadline=None):
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
            remaining = np.delete(best_perm, i)
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                new_perm = np.insert(remaining, pos, job)
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    break
    
    return best_perm


def _swap_ls(perm, pt, max_iters=50, deadline=None):
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
                    break
            if improved:
                break
    
    return best_perm


def _2opt_ls(perm, pt, max_iters=50, deadline=None):
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
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                new_ms = _makespan_fast(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return best_perm


def _sa_with_insertion(perm, pt, deadline, n_restarts=8, rng=None):
    """Simulated annealing with insertion local search polish at each restart."""
    n = len(perm)
    if n <= 1:
        return perm
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    best_perm = perm.copy()
    best_ms = _makespan_fast(best_perm, pt)
    
    for restart in range(n_restarts):
        if time.time() >= deadline:
            break
        
        if restart == 0:
            current_perm = best_perm.copy()
        else:
            # Perturbation: random swap of 2-3 jobs from best solution
            current_perm = best_perm.copy()
            n_perturb = min(n, rng.integers(2, 4))
            for _ in range(n_perturb):
                i, j = rng.choice(n, 2, replace=False)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        current_ms = _makespan_fast(current_perm, pt)
        
        # SA parameters
        T = max(current_ms * 0.1, 1.0)
        cooling_rate = 0.95
        min_T = 1e-6
        max_iterations = 300
        
        for it in range(max_iterations):
            if time.time() >= deadline:
                break
            if T < min_T:
                break
            
            # Generate neighbor by random swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_ms = _makespan_fast(new_perm, pt)
            
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_perm = current_perm.copy()
                    best_ms = current_ms
            
            T *= cooling_rate
        
        # Insertion local search polish after each SA run
        if time.time() < deadline:
            polished = _insertion_ls(current_perm, pt, max_iters=30, deadline=deadline)
            pol_ms = _makespan_fast(polished, pt)
            if pol_ms < best_ms:
                best_perm = polished
                best_ms = pol_ms
        
        # Also try 2-opt on the polished solution
        if time.time() < deadline:
            polished2 = _2opt_ls(best_perm, pt, max_iters=20, deadline=deadline)
            pol2_ms = _makespan_fast(polished2, pt)
            if pol2_ms < best_ms:
                best_perm = polished2
                best_ms = pol2_ms
    
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
    
    # Local search with insertion
    if time.time() < deadline:
        ls_perm = _insertion_ls(best_perm, pt, max_iters=80, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try swap local search
    if time.time() < deadline:
        swp_perm = _swap_ls(best_perm, pt, max_iters=50, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Try 2-opt
    if time.time() < deadline:
        opt_perm = _2opt_ls(best_perm, pt, max_iters=50, deadline=deadline)
        opt_ms = _makespan_fast(opt_perm, pt)
        if opt_ms < best_ms:
            best_perm = opt_perm
            best_ms = opt_ms
    
    # SA with insertion polish - main search phase
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        sa_perm = _sa_with_insertion(best_perm, pt, deadline, n_restarts=10, rng=rng)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts with local search
    if time.time() < deadline:
        num_restarts = 10
        for r in range(num_restarts):
            if time.time() >= deadline:
                break
            random_perm = rng.permutation(n).astype(np.int64)
            # Quick insertion local search from random start
            ls_r = _insertion_ls(random_perm, pt, max_iters=30, deadline=deadline)
            ls_ms_r = _makespan_fast(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
            # If improved, try swap and 2-opt as well
            if ls_ms_r < best_ms and time.time() < deadline:
                ls_r2 = _swap_ls(ls_r, pt, max_iters=30, deadline=deadline)
                ls_ms_r2 = _makespan_fast(ls_r2, pt)
                if ls_ms_r2 < best_ms:
                    best_perm = ls_r2
                    best_ms = ls_ms_r2
            if time.time() < deadline:
                ls_r3 = _2opt_ls(best_perm, pt, max_iters=20, deadline=deadline)
                ls_ms_r3 = _makespan_fast(ls_r3, pt)
                if ls_ms_r3 < best_ms:
                    best_perm = ls_r3
                    best_ms = ls_ms_r3
    
    # Final polish with insertion local search
    if time.time() < deadline:
        final_perm = _insertion_ls(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with swap local search
    if time.time() < deadline:
        final_perm2 = _swap_ls(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    # Final polish with 2-opt
    if time.time() < deadline:
        final_perm3 = _2opt_ls(best_perm, pt, max_iters=50, deadline=deadline)
        final_ms3 = _makespan_fast(final_perm3, pt)
        if final_ms3 < best_ms:
            best_perm = final_perm3
            best_ms = final_ms3
    
    return list(best_perm)
# EVOLVE-BLOCK-END
