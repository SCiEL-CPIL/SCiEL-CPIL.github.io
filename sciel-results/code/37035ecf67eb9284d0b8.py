"""Combined program: NEH/CDS variants + Matrix-based VNS local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    if n_machines <= 10:
        first_job = order[0]
        row = P[first_job]
        curr_row = [0.0] * n_machines
        curr_row[0] = row[0]
        for m in range(1, n_machines):
            curr_row[m] = curr_row[m-1] + row[m]
        for idx in range(1, n_jobs):
            j = order[idx]
            row = P[j]
            new_row = [0.0] * n_machines
            new_row[0] = curr_row[0] + row[0]
            for m in range(1, n_machines):
                prev_val = curr_row[m]
                left_val = new_row[m-1]
                new_row[m] = (prev_val if prev_val > left_val else left_val) + row[m]
            curr_row = new_row
        return curr_row[-1]
    else:
        first_job = order[0]
        curr_row = np.cumsum(P[first_job])
        for idx in range(1, n_jobs):
            j = order[idx]
            prev_row = curr_row
            new_row = np.empty(n_machines)
            new_row[0] = prev_row[0] + P[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    new_row[m] = max(prev_row[m], new_row[m-1]) + P[j, m]
            curr_row = new_row
        return curr_row[-1]

def _completion_matrix(order, P):
    """Return completion-time matrix C where C[i, m] is completion of order[i] on machine m."""
    n_jobs = len(order)
    if n_jobs == 0:
        return np.empty((0, P.shape[1]), dtype=float)
    n_machines = P.shape[1]
    C = np.empty((n_jobs, n_machines), dtype=float)
    first = order[0]
    C[0, 0] = P[first, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + P[first, m]
    for i in range(1, n_jobs):
        j = order[i]
        C[i, 0] = C[i-1, 0] + P[j, 0]
        for m in range(1, n_machines):
            C[i, m] = max(C[i-1, m], C[i, m-1]) + P[j, m]
    return C

def _makespan_from_C(C):
    return C[-1, -1]

def _eval_swap(order, C, P, i):
    """Evaluate swapping positions i and i+1. Returns new C and makespan."""
    n_jobs = len(order)
    n_machines = P.shape[1]
    if i < 0 or i >= n_jobs - 1:
        return C, _makespan_from_C(C)
    
    new_order = list(order)
    new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
    
    # Build new completion matrix
    new_C = np.empty((n_jobs, n_machines), dtype=float)
    # Copy unaffected prefix
    if i > 0:
        new_C[:i] = C[:i]
    
    # Recompute from position i onward
    for idx in range(i, n_jobs):
        j = new_order[idx]
        if idx == 0:
            new_C[0, 0] = P[j, 0]
            for m in range(1, n_machines):
                new_C[0, m] = new_C[0, m-1] + P[j, m]
        else:
            new_C[idx, 0] = new_C[idx-1, 0] + P[j, 0]
            for m in range(1, n_machines):
                new_C[idx, m] = max(new_C[idx-1, m], new_C[idx, m-1]) + P[j, m]
    
    return new_C, new_C[-1, -1]

def _eval_relocate(order, C, P, i, j, block_len=1):
    """
    Move block of block_len jobs from position i to position j.
    j is the target position in the NEW sequence (after removal).
    Returns new C and makespan.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if i < 0 or i + block_len > n_jobs:
        return C, _makespan_from_C(C)
    
    # Extract block
    block = list(order[i:i+block_len])
    # Remove block
    remaining = list(order[:i] + order[i+block_len:])
    # Adjust j: if j was after the removed block, shift left
    if j > i:
        adj_j = j - block_len
    else:
        adj_j = j
    
    if adj_j < 0:
        adj_j = 0
    if adj_j > len(remaining):
        adj_j = len(remaining)
    
    # Insert block at adj_j
    new_order = remaining[:adj_j] + block + remaining[adj_j:]
    
    # Build new completion matrix
    new_C = np.empty((n_jobs, n_machines), dtype=float)
    for idx in range(n_jobs):
        j_idx = new_order[idx]
        if idx == 0:
            new_C[0, 0] = P[j_idx, 0]
            for m in range(1, n_machines):
                new_C[0, m] = new_C[0, m-1] + P[j_idx, m]
        else:
            new_C[idx, 0] = new_C[idx-1, 0] + P[j_idx, 0]
            for m in range(1, n_machines):
                new_C[idx, m] = max(new_C[idx-1, m], new_C[idx, m-1]) + P[j_idx, m]
    
    return new_C, new_C[-1, -1]

def _eval_reverse(order, C, P, start, length):
    """Reverse a block of length 'length' starting at position 'start'."""
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if start < 0 or start + length > n_jobs or length < 2:
        return C, _makespan_from_C(C)
    
    new_order = list(order)
    new_order[start:start+length] = reversed(new_order[start:start+length])
    
    # Build new completion matrix
    new_C = np.empty((n_jobs, n_machines), dtype=float)
    for idx in range(n_jobs):
        j_idx = new_order[idx]
        if idx == 0:
            new_C[0, 0] = P[j_idx, 0]
            for m in range(1, n_machines):
                new_C[0, m] = new_C[0, m-1] + P[j_idx, m]
        else:
            new_C[idx, 0] = new_C[idx-1, 0] + P[j_idx, 0]
            for m in range(1, n_machines):
                new_C[idx, m] = max(new_C[idx-1, m], new_C[idx, m-1]) + P[j_idx, m]
    
    return new_C, new_C[-1, -1]

def _neh_heuristic(P, sort_key='total', seed=None):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    if sort_key == 'total':
        totals = P.sum(axis=1)
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            totals = totals + noise
        sorted_indices = np.argsort(-totals)
    elif sort_key == 'first_last':
        key_vals = P[:, 0] + P[:, -1]
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'first':
        key_vals = P[:, 0].copy()
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    elif sort_key == 'last':
        key_vals = P[:, -1].copy()
        if seed is not None:
            rng = np.random.RandomState(seed)
            noise = rng.uniform(0, 1e-6, n_jobs)
            key_vals = key_vals + noise
        sorted_indices = np.argsort(-key_vals)
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)

    current_seq = [int(sorted_indices[0])]
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq

def _cds_heuristic(P, seed=None):
    """CDS heuristic: starts from last machine, inserts in reverse order."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    totals = P.sum(axis=1)
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.uniform(0, 1e-6, n_jobs)
        totals = totals + noise
    sorted_indices = np.argsort(-totals)

    current_seq = [int(sorted_indices[-1])]
    for k in range(n_jobs - 2, -1, -1):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq

def _priority_heuristic(P, seed=None):
    """Priority-based heuristic using weighted machine priorities."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Weight machines by their position (later machines matter more for makespan)
    weights = np.linspace(1.0, 2.0, n_machines)
    priority = P @ weights
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.uniform(0, 1e-6, n_jobs)
        priority = priority + noise
    order = list(np.argsort(-priority))
    
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm

def _johnsons_generalized(P, seed=None):
    """Generalized Johnson's rule for multi-machine flow shop."""
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    if n_machines <= 2:
        if n_machines == 2:
            left = []
            right = []
            for j in range(n_jobs):
                if P[j, 0] <= P[j, 1]:
                    left.append((P[j, 0], j))
                else:
                    right.append((P[j, 1], j))
            left.sort(key=lambda x: x[0])
            right.sort(key=lambda x: -x[0])
            return [x[1] for x in left] + [x[1] for x in right]
        else:
            return list(np.argsort(P[:, 0]))
    
    # For multi-machine: use sum of first half and second half
    mid = n_machines // 2
    a = P[:, :mid].sum(axis=1)
    b = P[:, mid:].sum(axis=1)
    
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.uniform(0, 1e-6, n_jobs)
        a = a + noise
    
    left = []
    right = []
    for j in range(n_jobs):
        if a[j] <= b[j]:
            left.append((a[j], j))
        else:
            right.append((b[j], j))
    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: -x[0])
    return [x[1] for x in left] + [x[1] for x in right]

def _local_search_vns(order, P, deadline=None):
    """
    Variable Neighborhood Search using completion-matrix-based move evaluation.
      N1: adjacent swaps
      N2: block reversals (length 2..k)
      N3: Or-opt relocations (move block of size 1..3)
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_C = _completion_matrix(current_order, P)
    current_ms = _makespan_from_C(current_C)
    n_machines = P.shape[1]

    max_cycles = 200

    for cycle in range(max_cycles):
        if deadline is not None and time.time() > deadline:
            break

        # N1: adjacent swaps - best-improvement
        best_swap_pos = -1
        best_swap_C = None
        best_swap_ms = current_ms
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_C, new_ms = _eval_swap(current_order, current_C, P, i)
            if new_ms < best_swap_ms - 1e-12:
                best_swap_ms = new_ms
                best_swap_pos = i
                best_swap_C = new_C
        if best_swap_pos >= 0:
            new_order = list(current_order)
            new_order[best_swap_pos], new_order[best_swap_pos+1] = new_order[best_swap_pos+1], new_order[best_swap_pos]
            current_order = new_order
            current_C = best_swap_C
            current_ms = best_swap_ms
            continue

        # N2: block reversals - best-improvement
        max_block = min(8, n_jobs)
        best_rev_start = -1
        best_rev_len = -1
        best_rev_C = None
        best_rev_ms = current_ms
        for block_len in range(2, max_block + 1):
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n_jobs - block_len + 1):
                if deadline is not None and time.time() > deadline:
                    break
                new_C, new_ms = _eval_reverse(current_order, current_C, P, i, block_len)
                if new_ms < best_rev_ms - 1e-12:
                    best_rev_ms = new_ms
                    best_rev_start = i
                    best_rev_len = block_len
                    best_rev_C = new_C
        if best_rev_start >= 0:
            new_order = list(current_order)
            new_order[best_rev_start:best_rev_start+best_rev_len] = reversed(new_order[best_rev_start:best_rev_start+best_rev_len])
            current_order = new_order
            current_C = best_rev_C
            current_ms = best_rev_ms
            continue

        # N3: Or-opt relocations - best-improvement
        best_oropt = None
        best_oropt_C = None
        best_oropt_ms = current_ms
        for block_size in [1, 2, 3]:
            if n_jobs < block_size + 1:
                continue
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n_jobs - block_size + 1):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(n_jobs - block_size + 1):
                    if j == i:
                        continue
                    # Skip trivial moves (inserting at same or adjacent position)
                    if j == i + block_size:
                        continue
                    new_C, new_ms = _eval_relocate(current_order, current_C, P, i, j, block_size)
                    if new_ms < best_oropt_ms - 1e-12:
                        best_oropt_ms = new_ms
                        best_oropt = (i, j, block_size)
                        best_oropt_C = new_C
        if best_oropt is not None:
            i, j, block_size = best_oropt
            block = list(current_order[i:i+block_size])
            remaining = list(current_order[:i] + current_order[i+block_size:])
            if j > i:
                adj_j = j - block_size
            else:
                adj_j = j
            if adj_j < 0:
                adj_j = 0
            if adj_j > len(remaining):
                adj_j = len(remaining)
            new_order = remaining[:adj_j] + block + remaining[adj_j:]
            current_order = new_order
            current_C = best_oropt_C
            current_ms = best_oropt_ms
            continue

        # No improvement: local optimum
        break

    return current_order

def _final_polish(order, P, deadline=None):
    """
    Final high-precision polish: exhaustive adjacent swaps and small relocations
    with very tight tolerance to squeeze out last improvements.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_C = _completion_matrix(current_order, P)
    current_ms = _makespan_from_C(current_C)
    
    # Multiple passes of exhaustive adjacent swaps
    for pass_num in range(10):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_C, new_ms = _eval_swap(current_order, current_C, P, i)
            if new_ms < current_ms - 1e-15:
                new_order = list(current_order)
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                current_order = new_order
                current_C = new_C
                current_ms = new_ms
                improved = True
        if not improved:
            break
    
    # Small relocations (move single jobs)
    for pass_num in range(5):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                if j == i or j == i + 1:
                    continue
                new_C, new_ms = _eval_relocate(current_order, current_C, P, i, j, 1)
                if new_ms < current_ms - 1e-15:
                    block = [current_order[i]]
                    remaining = list(current_order[:i] + current_order[i+1:])
                    adj_j = j - 1 if j > i else j
                    if adj_j < 0:
                        adj_j = 0
                    if adj_j > len(remaining):
                        adj_j = len(remaining)
                    new_order = remaining[:adj_j] + block + remaining[adj_j:]
                    current_order = new_order
                    current_C = new_C
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses multiple NEH/CDS/priority/Johnson variants followed by matrix-based VNS.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_perm = None
    best_ms = float('inf')

    initial_perms = []

    # NEH variants
    for sort_key in ['total', 'first_last', 'first', 'last']:
        if time.time() > deadline - 30.0:
            break
        perm = _neh_heuristic(P, sort_key=sort_key)
        ms = _makespan(perm, P)
        initial_perms.append((ms, perm))
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # CDS heuristic
    if time.time() < deadline - 30.0:
        perm_cds = _cds_heuristic(P)
        ms_cds = _makespan(perm_cds, P)
        initial_perms.append((ms_cds, perm_cds))
        if ms_cds < best_ms:
            best_ms = ms_cds
            best_perm = perm_cds

    # Priority heuristic
    if time.time() < deadline - 30.0:
        perm_pri = _priority_heuristic(P)
        ms_pri = _makespan(perm_pri, P)
        initial_perms.append((ms_pri, perm_pri))
        if ms_pri < best_ms:
            best_ms = ms_pri
            best_perm = perm_pri

    # Generalized Johnson
    if time.time() < deadline - 30.0:
        perm_john = _johnsons_generalized(P)
        ms_john = _makespan(perm_john, P)
        initial_perms.append((ms_john, perm_john))
        if ms_john < best_ms:
            best_ms = ms_john
            best_perm = perm_john

    # Randomized NEH restarts
    rng = random.Random(12345)
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        seed = rng.randint(0, 10**9)
        perm_r = _neh_heuristic(P, sort_key='total', seed=seed)
        ms_r = _makespan(perm_r, P)
        initial_perms.append((ms_r, perm_r))
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r

    # Sort initial solutions by makespan (best first)
    initial_perms.sort(key=lambda x: x[0])

    # Apply VNS local search to the top initial solutions
    max_ls_attempts = min(5, len(initial_perms))
    ls_deadline = deadline - 5.0

    for idx in range(max_ls_attempts):
        if time.time() > ls_deadline:
            break
        ms_init, perm_init = initial_perms[idx]
        if ms_init > best_ms and idx > 0:
            continue

        improved_perm = _local_search_vns(perm_init, P, deadline=ls_deadline)
        improved_ms = _makespan(improved_perm, P)

        if improved_ms < best_ms:
            best_ms = improved_ms
            best_perm = improved_perm

    # Final polish pass
    if time.time() < deadline - 1.0 and best_perm is not None:
        if time.time() < deadline - 2.0:
            final_perm = _local_search_vns(best_perm, P, deadline=deadline - 1.5)
            final_ms = _makespan(final_perm, P)
            if final_ms < best_ms:
                best_ms = final_ms
                best_perm = final_perm
        
        if time.time() < deadline - 0.3 and best_perm is not None:
            polished_perm = _final_polish(best_perm, P, deadline=deadline - 0.3)
            polished_ms = _makespan(polished_perm, P)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm

    return list(best_perm)

# EVOLVE-BLOCK-END
