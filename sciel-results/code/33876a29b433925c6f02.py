"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0

    completion = [0.0] * n_machines
    pt = processing_times
    for job_idx in perm:
        row = pt[job_idx]
        for m in range(n_machines):
            if m == 0:
                completion[m] += row[m]
            else:
                prev = completion[m - 1]
                cur = completion[m]
                completion[m] = (prev if prev > cur else cur) + row[m]
    return completion[-1]


def _compute_completion(perm, processing_times):
    """Compute the full C[j][m] completion-time matrix for a permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt = processing_times
    for j, job_idx in enumerate(perm):
        row = pt[job_idx]
        for m in range(n_machines):
            if m == 0:
                C[j, m] = (C[j - 1, m] if j > 0 else 0.0) + row[m]
            else:
                C[j, m] = max(C[j, m - 1], C[j - 1, m]) + row[m]
    return C


def _eval_swap(perm, C, i, processing_times):
    """
    Evaluate makespan after swapping adjacent jobs at positions i and i+1.
    Returns the new makespan (float). Uses incremental recomputation.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if i < 0 or i >= n_jobs - 1:
        return C[n_jobs - 1, n_machines - 1]

    pt = processing_times
    job_a = perm[i]
    job_b = perm[i + 1]

    # C before position i
    C_prev = C[i - 1] if i > 0 else np.zeros(n_machines)

    # New completion for job_b at position i
    C_new_i = np.zeros(n_machines)
    row_b = pt[job_b]
    for m in range(n_machines):
        if m == 0:
            C_new_i[m] = C_prev[m] + row_b[m]
        else:
            C_new_i[m] = max(C_new_i[m - 1], C_prev[m]) + row_b[m]

    # New completion for job_a at position i+1
    C_new_ip1 = np.zeros(n_machines)
    row_a = pt[job_a]
    for m in range(n_machines):
        if m == 0:
            C_new_ip1[m] = C_new_i[m] + row_a[m]
        else:
            C_new_ip1[m] = max(C_new_ip1[m - 1], C_new_i[m]) + row_a[m]

    # If this is the last two jobs, makespan is C_new_ip1[-1]
    if i + 1 == n_jobs - 1:
        return C_new_ip1[n_machines - 1]

    # Otherwise, propagate from position i+2 onwards
    C_in = C_new_ip1  # completion at position i+1 in new order
    for j in range(i + 2, n_jobs):
        job = perm[j]
        row = pt[job]
        for m in range(n_machines):
            if m == 0:
                C_in[m] = C_in[m] + row[m]
            else:
                C_in[m] = max(C_in[m - 1], C_in[m]) + row[m]
    return C_in[n_machines - 1]


def _eval_move(perm, C, i, k, processing_times):
    """
    Evaluate makespan after removing job at position i and inserting at position k.
    (0 <= i < k <= n_jobs; after removal, insert before original index k,
    so the job ends up at position k-1 in the new list of length n_jobs.)
    Returns the new makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if i >= k:
        return C[n_jobs - 1, n_machines - 1]

    pt = processing_times
    job = perm[i]
    row = pt[job]

    # Build new permutation conceptually: remove i, then insert before k
    # New order: perm[0..i-1], perm[i+1..k-1], perm[i], perm[k..]
    # The moved job goes to position (k-1) in the new list (0-indexed).

    # Completion before position i (unchanged)
    C_before = C[i - 1] if i > 0 else np.zeros(n_machines)

    # Jobs from i+1 to k-1 shift left by one
    # We need to compute their new completions
    C_curr = C_before.copy()
    for j in range(i + 1, k):
        job_j = perm[j]
        row_j = pt[job_j]
        for m in range(n_machines):
            if m == 0:
                C_curr[m] = C_curr[m] + row_j[m]
            else:
                C_curr[m] = max(C_curr[m - 1], C_curr[m]) + row_j[m]

    # Now insert the moved job
    for m in range(n_machines):
        if m == 0:
            C_curr[m] = C_curr[m] + row[m]
        else:
            C_curr[m] = max(C_curr[m - 1], C_curr[m]) + row[m]

    # Then jobs from k onwards
    for j in range(k, n_jobs):
        job_j = perm[j]
        row_j = pt[job_j]
        for m in range(n_machines):
            if m == 0:
                C_curr[m] = C_curr[m] + row_j[m]
            else:
                C_curr[m] = max(C_curr[m - 1], C_curr[m]) + row_j[m]

    return C_curr[n_machines - 1]


def _neh_order(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []

    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)

    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]

    return perm


def _local_search(perm, processing_times, deadline, max_passes=200):
    """
    Local search using adjacent swaps and moves.
    Returns improved permutation.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm

    current_perm = list(perm)
    C = _compute_completion(current_perm, processing_times)
    current_ms = C[n_jobs - 1, processing_times.shape[1] - 1]

    for _pass in range(max_passes):
        if time.time() > deadline:
            break

        improved = False

        # Pass 1: adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                return current_perm
            new_ms = _eval_swap(current_perm, C, i, processing_times)
            if new_ms < current_ms - 1e-12:
                # Perform the swap
                current_perm[i], current_perm[i + 1] = current_perm[i + 1], current_perm[i]
                C = _compute_completion(current_perm, processing_times)
                current_ms = C[n_jobs - 1, processing_times.shape[1] - 1]
                improved = True
                break

        if improved:
            continue

        # Pass 2: moves (remove and reinsert)
        for i in range(n_jobs):
            if time.time() > deadline:
                return current_perm
            for k in range(i + 2, n_jobs):
                new_ms = _eval_move(current_perm, C, i, k, processing_times)
                if new_ms < current_ms - 1e-12:
                    # Perform the move
                    job = current_perm.pop(i)
                    current_perm.insert(k - 1, job)
                    C = _compute_completion(current_perm, processing_times)
                    current_ms = C[n_jobs - 1, processing_times.shape[1] - 1]
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search + perturbations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin: 145s out of 150s limit

    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    # Step 1: NEH constructive heuristic
    perm = _neh_order(processing_times)
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)

    # Step 2: Local search (swaps + moves)
    perm = _local_search(perm, processing_times, deadline)
    ms = _makespan(perm, processing_times)
    if ms < best_ms:
        best_perm = list(perm)
        best_ms = ms

    # Step 3: Perturbation + local search restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        perturbed = list(best_perm)

        # Or-opt style perturbation: pick a block of size 1-3 and move it
        block_size = min(rng.integers(1, 4), n_jobs)
        i = rng.integers(0, n_jobs - block_size + 1)
        block = perturbed[i:i + block_size]
        # Remove block
        perturbed = perturbed[:i] + perturbed[i + block_size:]
        # Insert at random position
        k = rng.integers(0, len(perturbed) + 1)
        perturbed = perturbed[:k] + block + perturbed[k:]

        # Local search from perturbed start
        improved_perm = _local_search(perturbed, processing_times, deadline)
        improved_ms = _makespan(improved_perm, processing_times)
        if improved_ms < best_ms:
            best_perm = list(improved_perm)
            best_ms = improved_ms

    return best_perm
# EVOLVE-BLOCK-END
