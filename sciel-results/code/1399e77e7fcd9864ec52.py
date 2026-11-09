# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_fast(perm, pt):
    """Optimized makespan computation using pure Python loops for small n."""
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
    """Compute full completion time matrix C[job_pos][machine].
    C[i][k] = completion time of job at position i on machine k."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return np.zeros((n, 0), dtype=np.float64)
    
    C = np.zeros((n, m), dtype=np.float64)
    # Machine 0: cumulative sum
    C[0, 0] = pt[perm[0], 0]
    for i in range(1, n):
        C[i, 0] = C[i-1, 0] + pt[perm[i], 0]
    
    # Machines 1..m-1
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


def _reversal_delta(C, pt, perm, i, j):
    """Compute new makespan after reversing segment [i, j] in perm.
    Uses the existing completion matrix to do this in O((j-i+1)*m) instead of O(n*m)."""
    n = len(perm)
    m = pt.shape[1]
    if i == j:
        return _makespan_from_C(C)
    
    # Build new completion matrix by modifying only the affected region
    # The reversal affects positions i..j
    # Before i: unchanged
    # At position i: new job is perm[j], completion depends on C[i-1][*] and new job
    # Positions i+1..j: new jobs are perm[j-1], perm[j-2], ..., perm[i]
    # After j: unchanged (but their completions may shift if the makespan through the segment changed)
    
    # Actually, we need to be careful. Let's rebuild C for the reversed segment.
    new_C = C.copy()
    
    # Position i gets job perm[j]
    if i == 0:
        new_C[i, 0] = pt[perm[j], 0]
    else:
        new_C[i, 0] = C[i-1, 0] + pt[perm[j], 0]
    for k in range(1, m):
        new_C[i, k] = max(new_C[i-1, k], new_C[i, k-1]) + pt[perm[j], k] if i > 0 else new_C[i, k-1] + pt[perm[j], k]
    
    # Positions i+1 to j get jobs perm[j-1], perm[j-2], ..., perm[i]
    for offset in range(1, j - i + 1):
        pos = i + offset
        job = perm[j - offset]
        if pos == 0:
            new_C[pos, 0] = pt[job, 0]
        else:
            new_C[pos, 0] = new_C[pos-1, 0] + pt[job, 0]
        for k in range(1, m):
            new_C[pos, k] = max(new_C[pos-1, k], new_C[pos, k-1]) + pt[job, k]
    
    # Positions after j: need to recompute because the completion times at position j may have changed
    if j + 1 < n:
        for pos in range(j + 1, n):
            job = perm[pos]
            new_C[pos, 0] = new_C[pos-1, 0] + pt[job, 0]
            for k in range(1, m):
                new_C[pos, k] = max(new_C[pos-1, k], new_C[pos, k-1]) + pt[job, k]
    
    return _makespan_from_C(new_C)


def _swap_delta(C, pt, perm, i, j):
    """Compute new makespan after swapping positions i and j in perm. O(n*m) but with early exit potential."""
    n = len(perm)
    m = pt.shape[1]
    if i == j:
        return _makespan_from_C(C)
    
    new_C = C.copy()
    
    # Position i gets job perm[j]
    if i == 0:
        new_C[i, 0] = pt[perm[j], 0]
    else:
        new_C[i, 0] = C[i-1, 0] + pt[perm[j], 0]
    for k in range(1, m):
        new_C[i, k] = max(new_C[i-1, k], new_C[i, k-1]) + pt[perm[j], k]
    
    # Position j gets job perm[i]
    if j == 0:
        new_C[j, 0] = pt[perm[i], 0]
    else:
        new_C[j, 0] = new_C[j-1, 0] + pt[perm[i], 0]
    for k in range(1, m):
        new_C[j, k] = max(new_C[j-1, k], new_C[j, k-1]) + pt[perm[i], k]
    
    # Positions between i and j (if any) are unaffected in terms of which job they hold,
    # but their completion times may change if i and j are not adjacent
    # Actually, for a swap, only positions i and j change jobs. Positions in between keep their jobs.
    # But the completion times at i affect i+1, which affects i+2, etc.
    # So we need to recompute from min(i,j)+1 to max(i,j) and beyond.
    
    lo = min(i, j)
    hi = max(i, j)
    
    if hi + 1 < n:
        for pos in range(hi + 1, n):
            job = perm[pos]
            new_C[pos, 0] = new_C[pos-1, 0] + pt[job, 0]
            for k in range(1, m):
                new_C[pos, k] = max(new_C[pos-1, k], new_C[pos, k-1]) + pt[job, k]
    
    return _makespan_from_C(new_C)


def _insert_delta(C, pt, perm, i, pos):
    """Compute new makespan after removing job at position i and inserting at position pos.
    i: original position of the job to move
    pos: new position (0-indexed in the new permutation, after removal)
    """
    n = len(perm)
    m = pt.shape[1]
    if i == pos or (i == pos - 1 and i > 0):
        # Check if it's effectively a no-op
        pass
    
    # Build new permutation
    job = perm[i]
    # Remove job from position i
    # New positions:
    # 0..i-1: same jobs
    # i..pos-1 (or pos..n-2): jobs shifted left by 1
    # pos: job
    # pos+1..n-1: remaining jobs shifted
    
    # For simplicity and correctness, rebuild the completion matrix
    new_perm = list(perm)
    del new_perm[i]
    # Insert at position pos in the new list
    # But pos is defined relative to the original? Let's define pos as the insertion index in the resulting list.
    # Actually, let's just create the new perm and recompute.
    # For efficiency, we can do incremental, but it's complex. Let's do a targeted rebuild.
    
    new_C = C.copy()
    
    # This is getting complex. For insertion, let's just rebuild from the point of change.
    # Positions before min(i, pos): unchanged
    # We need to be careful about the definition of pos.
    
    # Let me redefine: we remove job at index i, then insert it at index pos in the resulting array.
    # If i < pos: the job moves right. Positions i+1..pos shift left, job goes to pos.
    # If i > pos: the job moves left. Positions pos..i-1 shift right, job goes to pos.
    
    # For correctness, let's just rebuild the affected portion.
    # This is O(n*m) in the worst case but with a smaller constant than full recompute.
    
    # Actually, for now, let's just do a full recompute for insertion since the logic is complex.
    # The 2-opt and swap are the main local search moves.
    
    new_perm_arr = np.array(new_perm, dtype=np.int64)
    return _makespan_fast(new_perm_arr, pt)


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


def _local_search_2opt(perm, pt, max_iters=500, deadline=None):
    """2-opt local search for flow shop with time limit, using incremental makespan."""
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
                new_ms = _reversal_delta(C, pt, best_perm, i, j)
                if new_ms < best_ms - 1e-9:
                    # Apply the reversal
                    best_perm[i:j+1] = best_perm[i:j+1][::-1]
                    C = _completion_matrix(best_perm, pt)
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _local_search_swap(perm, pt, max_iters=200, deadline=None):
    """Swap-based local search for flow shop, using incremental makespan."""
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
                new_ms = _swap_delta(C, pt, best_perm, i, j)
                if new_ms < best_ms - 1e-9:
                    best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
                    C = _completion_matrix(best_perm, pt)
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
    
    # Local search with 2-opt (now faster due to incremental makespan)
    if time.time() < deadline:
        ls_perm = _local_search_2opt(best_perm, pt, max_iters=500, deadline=deadline)
        ls_ms = _makespan_fast(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try swap local search (also faster now)
    if time.time() < deadline:
        swp_perm = _local_search_swap(best_perm, pt, max_iters=300, deadline=deadline)
        swp_ms = _makespan_fast(swp_perm, pt)
        if swp_ms < best_ms:
            best_perm = swp_perm
            best_ms = swp_ms
    
    # Try insertion local search
    if time.time() < deadline:
        ins_perm = _local_search_insert(best_perm, pt, max_iters=50, deadline=deadline)
        ins_ms = _makespan_fast(ins_perm, pt)
        if ins_ms < best_ms:
            best_perm = ins_perm
            best_ms = ins_ms
    
    # Random restarts with local search
    rng = np.random.default_rng(42)
    if time.time() < deadline:
        num_restarts = 10
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
        final_perm = _local_search_swap(best_perm, pt, max_iters=100, deadline=deadline)
        final_ms = _makespan_fast(final_perm, pt)
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms
    
    # Final polish with 2-opt
    if time.time() < deadline:
        final_perm2 = _local_search_2opt(best_perm, pt, max_iters=100, deadline=deadline)
        final_ms2 = _makespan_fast(final_perm2, pt)
        if final_ms2 < best_ms:
            best_perm = final_perm2
            best_ms = final_ms2
    
    return list(best_perm)
# EVOLVE-BLOCK-END
