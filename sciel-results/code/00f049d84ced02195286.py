"""Combined program: fast NEH variants with efficient local search."""
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
        
    # Use list-based approach for small n_machines to reduce NumPy overhead
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
    if n_machines == 1:
        C[0, 0] = P[first, 0]
    else:
        C[0, 0] = P[first, 0]
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + P[first, m]
    for i in range(1, n_jobs):
        j = order[i]
        C[i, 0] = C[i-1, 0] + P[j, 0]
        for m in range(1, n_machines):
            C[i, m] = max(C[i-1, m], C[i, m-1]) + P[j, m]
    return C

def _neh_heuristic(P, sort_key='total', seed=None):
    """
    NEH heuristic for flow shop scheduling.
    sort_key: 'total' (sum), 'first_last' (P[:,0] + P[:,-1])
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: Sort by specified key
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
    else:
        totals = P.sum(axis=1)
        sorted_indices = np.argsort(-totals)
        
    # Start with the first job
    current_seq = [int(sorted_indices[0])]
    
    # Insert remaining jobs one by one
    for k in range(1, n_jobs):
        job = int(sorted_indices[k])
        best_pos = -1
        best_makespan = float('inf')
        
        # Try all possible insertion positions (0 to len(current_seq))
        for pos in range(len(current_seq) + 1):
            # Create new sequence
            new_seq = current_seq[:pos] + [job] + current_seq[pos:]
            # Calculate makespan
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        # Insert at best position
        current_seq.insert(best_pos, job)
        
    return current_seq

def _makespan_fast(order, P):
    """Fast makespan computation using Python lists for small machine counts."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0

    # Pre-extract processing times as list of lists for speed
    P_list = P.tolist()

    first_job = order[0]
    row = P_list[first_job]
    curr = [0.0] * n_machines
    curr[0] = row[0]
    for m in range(1, n_machines):
        curr[m] = curr[m-1] + row[m]

    for idx in range(1, n_jobs):
        j = order[idx]
        row = P_list[j]
        new_curr = [0.0] * n_machines
        new_curr[0] = curr[0] + row[0]
        for m in range(1, n_machines):
            prev = curr[m]
            left = new_curr[m-1]
            new_curr[m] = (prev if prev > left else left) + row[m]
        curr = new_curr

    return curr[-1]

def _local_search(order, P, deadline=None):
    """
    Local search using Or-opt moves (blocks of size 1, 2, 3) with First Improvement.
    Uses fast makespan evaluation for candidate moves.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan_fast(current_order, P)

    # Or-opt: remove a block of size k (1, 2, 3) and re-insert it
    block_sizes = [1, 2, 3]

    # We will loop until no improvement or deadline
    while True:
        if deadline is not None and time.time() > deadline:
            break

        found_improvement = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break

            for k in block_sizes:
                if i + k > n_jobs:
                    continue

                block = current_order[i:i+k]
                # Construct rest
                if i == 0:
                    rest = current_order[i+k:]
                elif i+k == n_jobs:
                    rest = current_order[:i]
                else:
                    rest = current_order[:i] + current_order[i+k:]

                # Try all insertion positions in rest
                for j in range(len(rest) + 1):
                    if deadline is not None and time.time() > deadline:
                        break

                    # Skip if inserting back at same position
                    if j == i and k == 1:
                        continue
                    if j == i and k > 1 and j + k == i + k:
                        continue

                    new_order = rest[:j] + block + rest[j:]
                    new_ms = _makespan_fast(new_order, P)

                    if new_ms < current_ms:
                        current_order = new_order
                        current_ms = new_ms
                        found_improvement = True
                        break
                else:
                    continue
                break
            else:
                continue
            break

        if not found_improvement:
            break

    return current_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    Uses multiple NEH variants followed by local search.
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
    # Keep 15s safety margin before the 150s kill
    deadline = start_time + 135.0

    # Try multiple NEH variants and pick the best initial solution
    best_perm = None
    best_ms = float('inf')

    # Variant 1: sort by total processing time
    perm1 = _neh_heuristic(P, sort_key='total')
    ms1 = _makespan(perm1, P)
    if ms1 < best_ms:
        best_ms = ms1
        best_perm = perm1

    # Variant 2: sort by first + last machine processing time
    perm2 = _neh_heuristic(P, sort_key='first_last')
    ms2 = _makespan(perm2, P)
    if ms2 < best_ms:
        best_ms = ms2
        best_perm = perm2

    # Add randomized NEH restarts if time permits
    rng = random.Random(12345)
    max_restarts = 20
    for r in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        seed = rng.randint(0, 10**9)
        perm_r = _neh_heuristic(P, sort_key='total', seed=seed)
        ms_r = _makespan_fast(perm_r, P)
        if ms_r < best_ms:
            best_ms = ms_r
            best_perm = perm_r

    # Check if we have time for local search
    if time.time() < deadline - 10.0 and len(best_perm) > 1:
        # Apply local search to the best initial solution
        improved_perm = _local_search(best_perm, P, deadline=deadline - 5.0)
        improved_ms = _makespan_fast(improved_perm, P)

        if improved_ms < best_ms:
            best_perm = improved_perm
            best_ms = improved_ms

        # Second pass: try local search on second-best initial solution if time permits
        # and it's different from the best
        if time.time() < deadline - 8.0 and len(best_perm) > 1:
            # Collect all evaluated permutations
            all_perms = []
            # Re-evaluate NEH variants
            p1 = _neh_heuristic(P, sort_key='total')
            p2 = _neh_heuristic(P, sort_key='first_last')
            for p in [p1, p2]:
                ms_p = _makespan_fast(p, P)
                if p != best_perm and ms_p < best_ms + 0.01 * best_ms:
                    improved2 = _local_search(p, P, deadline=deadline - 3.0)
                    ms2 = _makespan_fast(improved2, P)
                    if ms2 < best_ms:
                        best_perm = improved2
                        best_ms = ms2
                    break

    return list(best_perm)

# EVOLVE-BLOCK-END
