"""Rewritten flow-shop heuristic: Enhanced NEH, Fast Incremental Local Search, and SA with Restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _makespan(perm, processing_times):
    """Compute makespan for a permutation of job indices."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0

    prev = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                # cur[m] = max(prev[m], cur[m-1]) + pt[m]
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prev = cur
    return float(prev[-1])

def _prefix_completions(perm, processing_times):
    """Return list of n_jobs+1 completion vectors for prefixes of perm."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix = [None] * (n_jobs + 1)
    prefix[0] = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        pt = processing_times[perm[j]]
        prev = prefix[j]
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = prev[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur[m] = (prev[m] if prev[m] > cur[m - 1] else cur[m - 1]) + pt[m]
        prefix[j + 1] = cur
    return prefix

def _eval_swap(perm, i, prefix, processing_times):
    """
    Makespan after swapping perm[i] and perm[i+1].
    Uses prefix[i] as the start state.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    C_start = prefix[i]
    pt_a = processing_times[perm[i]]
    pt_b = processing_times[perm[i+1]]
    
    # Job b first
    C_b = np.empty(n_machines, dtype=np.float64)
    C_b[0] = C_start[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = (C_start[m] if C_start[m] > C_b[m - 1] else C_b[m - 1]) + pt_b[m]
        
    # Job a second
    C_a = np.empty(n_machines, dtype=np.float64)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = (C_b[m] if C_b[m] > C_a[m - 1] else C_a[m - 1]) + pt_a[m]
        
    # Continue with rest
    cur = C_a
    for j in range(i + 2, n_jobs):
        pt = processing_times[perm[j]]
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
        cur = nxt
    return float(cur[-1])

def _eval_insert(perm, i, j, prefix, processing_times):
    """
    Makespan after moving perm[i] to position j.
    j is the index in the NEW list where the job will be placed.
    If i < j, the job moves right.
    If i > j, the job moves left.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return float(prefix[n_jobs][-1])
        
    job_to_move = perm[i]
    pt_job = processing_times[job_to_move]
    
    # Determine the range of jobs affected
    # If i < j: jobs i+1 to j-1 are shifted left. Job i moves to j.
    # If i > j: jobs j to i-1 are shifted right. Job i moves to j.
    
    # We reconstruct the completion times for the affected segment.
    # The prefix before the start of the affected segment is known.
    
    if i < j:
        # Segment to recompute: from index j to i (in new indices, it's i+1 to j-1 shifted, then job at j)
        # Actually, easier to just simulate the segment.
        # Start state is prefix[i] (state before job i in original, which is same as before job i+1 in new... wait)
        # Original: ... P[i] P[i+1] ... P[j] ...
        # New: ... P[i+1] ... P[j] P[i] ...
        # The state before P[i+1] in the new sequence is the same as the state before P[i] in the original? No.
        # In original, P[i] is at i. In new, P[i+1] is at i.
        # So the start state for the new segment is prefix[i] (completion after P[i-1]).
        
        C_start = prefix[i]
        
        # New sequence segment: P[i+1], P[i+2], ..., P[j], P[i]
        # Note: P[i] is the job being moved.
        
        cur = C_start
        # Process P[i+1] to P[j]
        for k in range(i + 1, j + 1):
            pt = processing_times[perm[k]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
            cur = nxt
        
        # Now process P[i] at position j
        nxt = np.empty(n_machines, dtype=np.float64)
        nxt[0] = cur[0] + pt_job[0]
        for m in range(1, n_machines):
            nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt_job[m]
        cur = nxt
        
        # Continue with rest P[j+1] to P[n-1]
        for k in range(j + 1, n_jobs):
            pt = processing_times[perm[k]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
            cur = nxt
            
    else: # i > j
        # Original: ... P[j] ... P[i] ...
        # New: ... P[i] P[j] ... P[i-1] ...
        # Start state is prefix[j] (completion after P[j-1]).
        
        C_start = prefix[j]
        
        # Process P[i] first
        cur = np.empty(n_machines, dtype=np.float64)
        cur[0] = C_start[0] + pt_job[0]
        for m in range(1, n_machines):
            cur[m] = (C_start[m] if C_start[m] > cur[m - 1] else C_start[m - 1]) + pt_job[m]
            
        # Process P[j] to P[i-1]
        for k in range(j, i):
            pt = processing_times[perm[k]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
            cur = nxt
            
        # Continue with rest P[i+1] to P[n-1]
        for k in range(i + 1, n_jobs):
            pt = processing_times[perm[k]]
            nxt = np.empty(n_machines, dtype=np.float64)
            nxt[0] = cur[0] + pt[0]
            for m in range(1, n_machines):
                nxt[m] = (cur[m] if cur[m] > nxt[m - 1] else nxt[m - 1]) + pt[m]
            cur = nxt

    return float(cur[-1])

def _neh(processing_times, rng, backward=False):
    """NEH constructive heuristic."""
    n_jobs, _ = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))

    totals = processing_times.sum(axis=1)
    # Randomize the order slightly to generate diversity? Or keep deterministic for baseline?
    # For diverse candidates, we can shuffle jobs with equal totals or just use different seeds for the whole process.
    order = list(np.argsort(-totals))
    
    if backward:
        order.reverse()

    partial = [order[0]]
    for k in range(1, n_jobs):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        # Try each insertion position
        for pos in range(len(partial) + 1):
            cand = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(cand, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]
        
    if backward:
        partial.reverse()
        
    return partial

def _local_search(perm, processing_times, deadline, rng):
    """Best-improvement 2-opt and insertion local search."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        prefix = _prefix_completions(best_perm, processing_times)
        
        # 1. Best-improvement 2-opt (adjacent swaps)
        best_i = -1
        best_swap_ms = best_ms
        for i in range(n_jobs - 1):
            ms = _eval_swap(best_perm, i, prefix, processing_times)
            if ms < best_swap_ms - 1e-12:
                best_swap_ms = ms
                best_i = i
        
        if best_i >= 0:
            best_perm[best_i], best_perm[best_i + 1] = best_perm[best_i + 1], best_perm[best_i]
            best_ms = best_swap_ms
            improved = True
            continue

        # 2. Best-improvement Insertion
        best_move = None
        best_ins_ms = best_ms
        
        # Limit insertion search if N is large to save time, but for typical sizes it's okay.
        # We check all i, j. Complexity O(N^2 * M).
        # To speed up, we can skip j == i and j == i+1 (swap) and j == i-1 (swap).
        
        for i in range(n_jobs):
            if time.time() > deadline - 1.0:
                break
            for j in range(n_jobs):
                if j == i or j == i + 1 or j == i - 1:
                    continue
                # j is the index in the NEW list.
                # If i < j, moving right.
                # If i > j, moving left.
                ms = _eval_insert(best_perm, i, j, prefix, processing_times)
                if ms < best_ins_ms - 1e-12:
                    best_ins_ms = ms
                    best_move = (i, j, ms)
        
        if best_move is not None:
            i, j, ms = best_move
            job = best_perm.pop(i)
            best_perm.insert(j, job)
            best_ms = ms
            improved = True

    return best_perm, best_ms

def _simulated_annealing(perm, ms, processing_times, deadline, rng):
    """SA with restarts."""
    n_jobs = len(perm)
    cur_perm = list(perm)
    cur_ms = ms
    best_perm = list(cur_perm)
    best_ms = cur_ms
    
    T = cur_ms * 0.1
    T_min = max(cur_ms * 0.001, 1e-6)
    cool = 0.995
    
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if T <= T_min:
            break
            
        prefix = _prefix_completions(cur_perm, processing_times)
        
        # Generate neighbor
        if rng.random() < 0.5:
            # Swap
            i = rng.randint(0, n_jobs - 2)
            new_ms = _eval_swap(cur_perm, i, prefix, processing_times)
            new_perm = list(cur_perm)
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
        else:
            # Insert
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1 or j == i - 1:
                T *= cool
                continue
            new_ms = _eval_insert(cur_perm, i, j, prefix, processing_times)
            new_perm = list(cur_perm)
            job = new_perm.pop(i)
            new_perm.insert(j, job)
            
        delta = new_ms - cur_ms
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_perm = new_perm
            cur_ms = new_ms
            if cur_ms < best_ms:
                best_ms = cur_ms
                best_perm = list(cur_perm)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        T *= cool
        
        # Restart logic
        if no_improve_steps > max_no_improve:
            # Perturb best solution and restart SA
            cur_perm = list(best_perm)
            cur_ms = best_ms
            # Kick: random swap
            for _ in range(3):
                i = rng.randint(0, n_jobs - 2)
                cur_perm[i], cur_perm[i + 1] = cur_perm[i + 1], cur_perm[i]
            cur_ms = _makespan(cur_perm, processing_times)
            T = best_ms * 0.1
            no_improve_steps = 0

    return best_perm, best_ms

def order_jobs(processing_times):
    start = time.time()
    deadline = start + 145.0  # Safety margin

    n_jobs, _ = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    rng = random.Random(123)

    # Generate diverse starting solutions
    candidates = []
    candidates.append(_neh(processing_times, rng, backward=False))
    candidates.append(_neh(processing_times, rng, backward=True))
    
    # Random permutations
    for _ in range(3):
        rand_perm = list(range(n_jobs))
        rng.shuffle(rand_perm)
        candidates.append(rand_perm)
        
    # LPT
    candidates.append(list(np.argsort(-processing_times.sum(axis=1))))

    best_perm = candidates[0]
    best_ms = _makespan(best_perm, processing_times)
    for p in candidates[1:]:
        ms = _makespan(p, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = p

    # Phase 1: Local Search
    if time.time() < deadline - 10.0:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 10.0, rng)

    # Phase 2: Simulated Annealing
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _simulated_annealing(best_perm, best_ms, processing_times, deadline - 5.0, rng)

    # Phase 3: Final Local Search Polish
    if time.time() < deadline - 1.0:
        best_perm, best_ms = _local_search(best_perm, processing_times, deadline - 1.0, rng)

    return list(best_perm)
# EVOLVE-BLOCK-END
