# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_full(pt, perm):
    """Full makespan computation. O(n*m)."""
    n, m = pt.shape
    if n == 0:
        return 0.0
    P = pt[perm]
    if m == 1:
        return float(P.sum())
    C = np.empty((n, m), dtype=np.float64)
    C[0, :] = np.cumsum(P[0, :])
    C[1:, 0] = C[:-1, 0] + P[1:, 0]
    for k in range(1, m):
        C[1:, k] = np.maximum(C[1:, k - 1], C[:-1, k]) + P[1:, k]
    return float(C[-1, -1])


def _compute_C(pt, perm):
    """Compute full completion time matrix C. Returns (C, makespan)."""
    n, m = pt.shape
    if n == 0:
        return np.empty((0, m)), 0.0
    P = pt[perm]
    if m == 1:
        C = np.cumsum(P, axis=0)[:, :1]
        return C, float(C[-1, 0])
    C = np.empty((n, m), dtype=np.float64)
    C[0, :] = np.cumsum(P[0, :])
    C[1:, 0] = C[:-1, 0] + P[1:, 0]
    for k in range(1, m):
        C[1:, k] = np.maximum(C[1:, k - 1], C[:-1, k]) + P[1:, k]
    return C, float(C[-1, -1])


def _makespan_after_2opt(pt, perm, C, a, b):
    """
    Compute new makespan after reversing segment [a:b] (inclusive) in perm.
    Only recomputes rows a..b of C, using C[a-1] as boundary.
    Returns (new_C, new_makespan).
    """
    n, m = pt.shape
    if a >= b:
        return C, float(C[-1, -1])
    
    P = pt[perm]
    new_C = C.copy()
    
    # Reversed segment
    seg = P[a:b+1][::-1]
    
    if a == 0:
        # No upper boundary from previous row
        new_C[a, :] = np.cumsum(seg[0, :])
        for i in range(1, b - a + 1):
            new_C[a + i, 0] = new_C[a + i - 1, 0] + seg[i, 0]
            for k in range(1, m):
                new_C[a + i, k] = max(new_C[a + i, k - 1], new_C[a + i - 1, k]) + seg[i, k]
    else:
        # Boundary from row a-1
        prev_row = C[a - 1, :]
        # First row of segment (was row b, now at position a)
        new_C[a, 0] = prev_row[0] + seg[0, 0]
        for k in range(1, m):
            new_C[a, k] = max(new_C[a, k - 1], prev_row[k]) + seg[0, k]
        for i in range(1, b - a + 1):
            new_C[a + i, 0] = new_C[a + i - 1, 0] + seg[i, 0]
            for k in range(1, m):
                new_C[a + i, k] = max(new_C[a + i, k - 1], new_C[a + i - 1, k]) + seg[i, k]
    
    # If b+1 < n, we need to check if rows after b change
    # Row b+1 depends on row b (which changed) and row b (old) -> actually row b+1 depends on new row b
    # But rows b+2..n-1 depend on row b+1, etc. So we need to propagate forward.
    if b + 1 < n:
        for i in range(b + 1, n):
            new_C[i, 0] = new_C[i - 1, 0] + P[i, 0]
            for k in range(1, m):
                new_C[i, k] = max(new_C[i, k - 1], new_C[i - 1, k]) + P[i, k]
    
    return new_C, float(new_C[-1, -1])


def _makespan_after_swap(pt, perm, C, i, j):
    """
    Compute new makespan after swapping positions i and j in perm (i < j).
    Only recomputes rows i..j of C.
    Returns (new_C, new_makespan).
    """
    n, m = pt.shape
    if i == j:
        return C, float(C[-1, -1])
    if i > j:
        i, j = j, i
    
    P = pt[perm]
    new_C = C.copy()
    
    # The segment i..j: positions i and j are swapped, positions i+1..j-1 stay in place
    # But their effective processing times change because the jobs at positions i and j are swapped.
    # Actually, the permutation changes: perm[i] and perm[j] swap.
    # So the processing time vector for positions i..j changes.
    
    # Build the processing times for positions i..j after swap
    seg_indices = list(range(i, j + 1))
    # After swap: position i has old perm[j], position j has old perm[i], rest unchanged
    seg_pt = np.empty((j - i + 1, m), dtype=np.float64)
    seg_pt[0, :] = P[j, :]  # position i gets job from position j
    for idx in range(1, j - i):
        seg_pt[idx, :] = P[i + idx, :]  # positions i+1..j-1 unchanged
    seg_pt[j - i, :] = P[i, :]  # position j gets job from position i
    
    if i == 0:
        new_C[i, :] = np.cumsum(seg_pt[0, :])
        for idx in range(1, j - i + 1):
            new_C[i + idx, 0] = new_C[i + idx - 1, 0] + seg_pt[idx, 0]
            for k in range(1, m):
                new_C[i + idx, k] = max(new_C[i + idx, k - 1], new_C[i + idx - 1, k]) + seg_pt[idx, k]
    else:
        prev_row = C[i - 1, :]
        new_C[i, 0] = prev_row[0] + seg_pt[0, 0]
        for k in range(1, m):
            new_C[i, k] = max(new_C[i, k - 1], prev_row[k]) + seg_pt[0, k]
        for idx in range(1, j - i + 1):
            new_C[i + idx, 0] = new_C[i + idx - 1, 0] + seg_pt[idx, 0]
            for k in range(1, m):
                new_C[i + idx, k] = max(new_C[i + idx, k - 1], new_C[i + idx - 1, k]) + seg_pt[idx, k]
    
    if j + 1 < n:
        for idx in range(j + 1, n):
            new_C[idx, 0] = new_C[idx - 1, 0] + P[idx, 0]
            for k in range(1, m):
                new_C[idx, k] = max(new_C[idx, k - 1], new_C[idx - 1, k]) + P[idx, k]
    
    return new_C, float(new_C[-1, -1])


def _neh_order(pt):
    """NEH heuristic: order by decreasing total processing time."""
    totals = pt.sum(axis=1)
    return np.argsort(-totals)


def _spt_order(pt):
    """SPT heuristic: order by increasing total processing time."""
    totals = pt.sum(axis=1)
    return np.argsort(totals)


def _lpt_order(pt):
    """LPT heuristic: order by decreasing average processing time."""
    avgs = pt.mean(axis=1)
    return np.argsort(-avgs)


def _max_machine_order(pt):
    """Order by decreasing max processing time on any single machine."""
    maxes = pt.max(axis=1)
    return np.argsort(-maxes)


def _constructive_priority(pt, priority_func, rng):
    """
    Greedy constructive heuristic: build schedule one job at a time.
    At each step, select the job that minimizes a surrogate cost.
    priority_func returns a score for each candidate job given current state.
    """
    n, m = pt.shape
    remaining = list(range(n))
    perm = np.empty(n, dtype=np.int64)
    
    # Track completion times on each machine
    machine_time = np.zeros(m, dtype=np.float64)
    
    for pos in range(n):
        best_job = -1
        best_cost = np.inf
        for job in remaining:
            # Estimate the makespan contribution if we add this job next
            # The job's completion on machine k is max(machine_time[k], C_prev_job_on_k-1) + pt[job,k]
            # Simplified: assume the job starts on machine 0 at machine_time[0]
            # and flows through machines
            # Cost = max over k of (machine_time[k] + sum of pt[job, k' for k' >= k] roughly)
            # Better: simulate the flow
            mt = machine_time.copy()
            for k in range(m):
                if k == 0:
                    mt[k] = machine_time[0] + pt[job, 0]
                else:
                    mt[k] = max(machine_time[k], mt[k-1]) + pt[job, k]
            # Cost is the max completion time
            cost = mt.max()
            if cost < best_cost:
                best_cost = cost
                best_job = job
        
        perm[pos] = best_job
        remaining.remove(best_job)
        # Update machine times
        job = best_job
        mt = machine_time.copy()
        for k in range(m):
            if k == 0:
                mt[k] = machine_time[0] + pt[job, 0]
            else:
                mt[k] = max(machine_time[k], mt[k-1]) + pt[job, k]
        machine_time = mt
    
    return perm


def _local_search_2opt(pt, perm, C, ms, deadline, rng, max_time_per_iter=0.001):
    """
    2-opt local search with incremental makespan updates.
    Tries random segment reversals and accepts improvements.
    """
    n, m = pt.shape
    if n <= 2:
        return perm, C, ms
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        # Try a batch of random 2-opt moves
        n_tries = max(50, int((deadline - 3.0 - time.time()) * 2000))
        n_tries = min(n_tries, 20000)
        
        for _ in range(n_tries):
            if time.time() > deadline - 3.0:
                break
            
            a = int(rng.integers(0, n - 1))
            b = int(rng.integers(0, n - 1))
            if a > b:
                a, b = b, a
            if b - a < 1:
                continue
            
            # Apply reversal
            perm[a:b+1] = perm[a:b+1][::-1]
            new_C, new_ms = _makespan_after_2opt(pt, perm, C, a, b)
            
            if new_ms < ms - 1e-12:
                ms = new_ms
                C = new_C
                improved = True
            else:
                # Revert
                perm[a:b+1] = perm[a:b+1][::-1]
    
    return perm, C, ms


def _local_search_swap(pt, perm, C, ms, deadline, rng):
    """
    Swap-based local search with incremental makespan updates.
    """
    n, m = pt.shape
    if n <= 1:
        return perm, C, ms
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        n_tries = max(100, int((deadline - 2.0 - time.time()) * 3000))
        n_tries = min(n_tries, 30000)
        
        for _ in range(n_tries):
            if time.time() > deadline - 2.0:
                break
            
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Apply swap
            perm[i], perm[j] = perm[j], perm[i]
            new_C, new_ms = _makespan_after_swap(pt, perm, C, i, j)
            
            if new_ms < ms - 1e-12:
                ms = new_ms
                C = new_C
                improved = True
            else:
                # Revert
                perm[i], perm[j] = perm[j], perm[i]
    
    return perm, C, ms


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
    if m == 1:
        return list(range(n))
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions
    initial_perms = []
    
    # 1. NEH
    initial_perms.append(_neh_order(pt).copy())
    # 2. SPT
    initial_perms.append(_spt_order(pt).copy())
    # 3. LPT
    initial_perms.append(_lpt_order(pt).copy())
    # 4. Max machine
    initial_perms.append(_max_machine_order(pt).copy())
    # 5. Constructive priority (greedy)
    initial_perms.append(_constructive_priority(pt, None, rng))
    # 6-8. Random permutations
    for _ in range(3):
        rp = np.random.default_rng(int(rng.integers(0, 2**31))).permutation(n)
        initial_perms.append(rp)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = np.inf
    best_C = None
    
    for perm in initial_perms:
        C, ms = _compute_C(pt, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
            best_C = C.copy()
    
    # Run local search from each initial solution, keep the best
    # Use remaining time budget
    remaining_time = deadline - time.time()
    n_starts = len(initial_perms)
    time_per_start = remaining_time / n_starts
    
    for idx, perm in enumerate(initial_perms):
        if time.time() > deadline - 5.0:
            break
        
        C, ms = _compute_C(pt, perm)
        
        # 2-opt local search
        time_budget_2opt = time.time() + min(time_per_start * 0.6, remaining_time * 0.5)
        perm, C, ms = _local_search_2opt(pt, perm, C, ms, time_budget_2opt, rng)
        
        # Swap local search
        time_budget_swap = time.time() + min(time_per_start * 0.4, remaining_time * 0.3)
        perm, C, ms = _local_search_swap(pt, perm, C, ms, time_budget_swap, rng)
        
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
            best_C = C.copy()
    
    # Final polish: run 2-opt from the best solution with remaining time
    if time.time() < deadline - 3.0:
        perm = best_perm.copy()
        C = best_C.copy()
        ms = best_ms
        perm, C, ms = _local_search_2opt(pt, perm, C, ms, deadline - 3.0, rng)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
