"""Flow-shop makespan minimization with incremental local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0

    pt = processing_times
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    prev = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        row = pt[perm[j]]
        curr = np.empty(n_machines, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            curr[m] = max(prev[m], curr[m - 1]) + row[m]
        prev = curr
    return float(prev[-1])


def _compute_completion(perm, processing_times):
    """Compute full completion matrix. Returns (makespan, comp_matrix)."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0, np.zeros((max(n_jobs, 1), max(n_machines, 1)))

    comp = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt = processing_times

    for j in range(n_jobs):
        row = pt[perm[j]]
        comp[j, 0] = (comp[j - 1, 0] if j > 0 else 0.0) + row[0]
        for m in range(1, n_machines):
            comp[j, m] = max(comp[j - 1, m] if j > 0 else 0.0, comp[j, m - 1]) + row[m]

    return float(comp[-1, -1]), comp


def _recompute_from(perm, processing_times, start_job, comp):
    """Recompute completion matrix from start_job onwards. Returns new makespan."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if start_job >= n_jobs:
        return float(comp[-1, -1]) if n_jobs > 0 else 0.0

    pt = processing_times
    # Recompute rows from start_job to end
    for j in range(start_job, n_jobs):
        row = pt[perm[j]]
        if j == 0:
            comp[j, 0] = row[0]
        else:
            comp[j, 0] = comp[j - 1, 0] + row[0]
        for m in range(1, n_machines):
            comp[j, m] = max(comp[j - 1, m], comp[j, m - 1]) + row[m]

    return float(comp[n_jobs - 1, n_machines - 1])


def _eval_swap(perm, processing_times, i, j, comp):
    """Evaluate makespan after swapping positions i and j. Returns new makespan.
    i < j assumed. Recomputes from min(i,j) onwards."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0

    # Swap in perm
    perm[i], perm[j] = perm[j], perm[i]
    # Recompute from min(i, j) onwards
    start = min(i, j)
    ms = _recompute_from(perm, processing_times, start, comp)
    # Swap back
    perm[i], perm[j] = perm[j], perm[i]
    return ms


def _eval_move(perm, processing_times, i, j, comp):
    """Evaluate makespan after moving job from position i to position j (i != j).
    j is the target index in the final permutation (0-indexed).
    Returns new makespan."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0

    job = perm[i]
    # Remove job from position i
    if i < j:
        # Elements i+1..j shift left by 1
        start_recompute = i
        # Build new perm without creating full list (we just track positions)
        # Actually for correctness, let's do it with a temporary array
        new_perm = perm.copy()
        # Remove from i
        new_perm[i:i+1] = new_perm[i+1:i+2]
        # Now insert at position j-1 (since removal shifted indices)
        # Wait, this is getting complex. Let me just use a simpler approach:
        # Create new permutation, compute, then restore
        pass

    # Simpler approach: create new perm, compute makespan from scratch for the affected region
    # But that's slow. Let me think differently.
    #
    # For a move from i to j:
    # If i < j: positions i+1, i+2, ..., j shift left by 1. New position of job is j-1.
    #   Affected region: from i to j (inclusive in original indexing)
    #   In new perm: positions i through j-1 contain the shifted jobs, position j-1 has the moved job
    #   Wait no. Let me re-think.
    #
    # Original: ... A_i, A_{i+1}, ..., A_{j-1}, A_j ...
    # Move A_i to position j (0-indexed target):
    # If i < j: ... A_{i+1}, A_{i+2}, ..., A_j, A_i ... (A_i goes to position j in new, but
    #   actually it replaces A_j and shifts? No, "insert at position j" means:
    #   Remove A_i, then insert before what was at position j (which is now at j-1 after removal)
    #   So final: ... A_{i+1}, ..., A_{j-1}, A_i, A_j ...
    #   Affected: positions i through j in new perm
    #
    # If i > j: ... A_j, ..., A_{i-1}, A_i, A_{i+1} ...
    #   Remove A_i, insert at position j:
    #   ... A_j, A_i, A_{j+1}, ..., A_{i-1}, A_{i+1} ...
    #   Wait, remove A_i first: positions j..i-1 stay, i..end shift left
    #   Then insert at j: positions j..i-1 shift right, A_i at position j
    #   Final: ... A_j, A_i, A_{j+1}, ..., A_{i-1}, A_{i+1} ...
    #   Affected: positions j through i in new perm

    # For simplicity and correctness, let me just build the new perm and recompute from the start point
    new_perm = list(perm)
    job = new_perm.pop(i)
    # After pop, if j > i, the target index shifts left by 1
    if j > i:
        new_perm.insert(j - 1, job)
    else:
        new_perm.insert(j, job)

    # Determine recompute start
    if i < j:
        start = i
    else:
        start = j

    # We need to recompute using the new permutation
    # Copy comp and recompute
    comp_new = comp.copy()
    n_m = n_machines
    pt = processing_times
    for jj in range(start, n_jobs):
        row = pt[new_perm[jj]]
        if jj == 0:
            comp_new[jj, 0] = row[0]
        else:
            comp_new[jj, 0] = comp_new[jj - 1, 0] + row[0]
        for m in range(1, n_m):
            comp_new[jj, m] = max(comp_new[jj - 1, m], comp_new[jj, m - 1]) + row[m]

    return float(comp_new[n_jobs - 1, n_machines - 1])


def _neh_order(processing_times):
    """NEH heuristic: sort by total processing time descending, insert each at best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))

    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]

    return partial


def _neh_forward_backward(processing_times):
    """NEH with forward and backward pass."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))

    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]

    # Backward pass
    result = [partial[-1]]
    for i in range(n_jobs - 2, -1, -1):
        job = partial[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(result) + 1):
            candidate = result[:pos] + [job] + result[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        result = result[:best_pos] + [job] + result[best_pos:]

    return result


def _local_search(perm, processing_times, deadline, use_insertion=True, use_swap=True, max_iterations=5000):
    """Local search with incremental completion matrix updates."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)

    perm = list(perm)
    ms, comp = _compute_completion(perm, processing_times)
    best_perm = list(perm)
    best_ms = ms
    iterations = 0

    while iterations < max_iterations and time.time() < deadline:
        improved = False
        iterations += 1

        if use_insertion:
            # Try moving each job to a different position
            for i in range(n_jobs):
                if time.time() >= deadline:
                    break
                for j in range(n_jobs):
                    if i == j:
                        continue
                    new_ms = _eval_move(perm, processing_times, i, j, comp)
                    if new_ms < best_ms - 1e-9:
                        # Apply the move
                        job = perm.pop(i)
                        if j > i:
                            perm.insert(j - 1, job)
                        else:
                            perm.insert(j, job)
                        best_ms = new_ms
                        # Recompute full comp
                        _, comp = _compute_completion(perm, processing_times)
                        best_perm = list(perm)
                        improved = True
                        break
                if improved:
                    break

        if not improved and use_swap:
            # Try adjacent swaps
            for i in range(n_jobs - 1):
                if time.time() >= deadline:
                    break
                new_ms = _eval_swap(perm, processing_times, i, i + 1, comp)
                if new_ms < best_ms - 1e-9:
                    perm[i], perm[i + 1] = perm[i + 1], perm[i]
                    best_ms = new_ms
                    _, comp = _compute_completion(perm, processing_times)
                    best_perm = list(perm)
                    improved = True
            if improved:
                continue

        if not improved:
            break

    return best_perm, best_ms


def _simulated_annealing(perm, processing_times, deadline, start_temp=10.0, cool_rate=0.995, max_iterations=20000):
    """Simulated annealing with incremental evaluation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return list(perm), _makespan(perm, processing_times)

    perm = list(perm)
    ms, comp = _compute_completion(perm, processing_times)
    best_perm = list(perm)
    best_ms = ms
    current_ms = ms
    temp = start_temp
    rng = random.Random(123)
    iterations = 0

    while iterations < max_iterations and time.time() < deadline and temp > 1e-6:
        iterations += 1

        # Choose a random move: swap or insertion
        move_type = rng.random()
        if move_type < 0.5:
            # Swap two random positions
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if i == j:
                continue
            new_ms = _eval_swap(perm, processing_times, i, j, comp)

            # Accept or reject
            delta = new_ms - current_ms
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                # Apply swap
                perm[i], perm[j] = perm[j], perm[i]
                current_ms = new_ms
                _, comp = _compute_completion(perm, processing_times)
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = list(perm)
        else:
            # Insertion: move random job to random position
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if i == j:
                continue
            new_ms = _eval_move(perm, processing_times, i, j, comp)

            delta = new_ms - current_ms
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                job = perm.pop(i)
                if j > i:
                    perm.insert(j - 1, job)
                else:
                    perm.insert(j, job)
                current_ms = new_ms
                _, comp = _compute_completion(perm, processing_times)
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = list(perm)

        temp *= cool_rate

    return best_perm, best_ms


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n_jobs, n_machines = processing_times.shape

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    # Phase 1: Generate initial solutions
    best_perm = None
    best_ms = float('inf')

    # Baseline: LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    lpt_ms = _makespan(lpt_perm, processing_times)
    if lpt_ms < best_ms:
        best_perm = lpt_perm
        best_ms = lpt_ms

    # NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times)
        neh_ms = _makespan(neh_perm, processing_times)
        if neh_ms < best_ms:
            best_perm = neh_perm
            best_ms = neh_ms

    # NEH forward-backward
    if time.time() < deadline:
        nf_perm = _neh_forward_backward(processing_times)
        nf_ms = _makespan(nf_perm, processing_times)
        if nf_ms < best_ms:
            best_perm = nf_perm
            best_ms = nf_ms

    # Phase 2: Multi-start local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        n_starts = max(1, min(5, int(remaining / 15.0)))
        rng = random.Random(42)

        for s in range(n_starts):
            if time.time() >= deadline - 5.0:
                break

            # Perturb the best solution
            start_perm = list(best_perm)
            # Apply some random swaps
            n_perturb = max(1, n_jobs // 5)
            for _ in range(n_perturb):
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                start_perm[i], start_perm[j] = start_perm[j], start_perm[i]

            ls_deadline = min(deadline - 5.0, start_time + (deadline - start_time) * 0.7)
            if s == n_starts - 1:
                ls_deadline = deadline - 2.0

            ls_perm, ls_ms = _local_search(
                start_perm, processing_times, ls_deadline,
                use_insertion=True, use_swap=True, max_iterations=3000
            )
            if ls_ms < best_ms:
                best_perm = ls_perm
                best_ms = ls_ms

    # Phase 3: Final polish with simulated annealing
    if time.time() < deadline - 3.0:
        sa_deadline = deadline - 1.0
        sa_perm, sa_ms = _simulated_annealing(
            best_perm, processing_times, sa_deadline,
            start_temp=max(1.0, best_ms * 0.01), cool_rate=0.997, max_iterations=50000
        )
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms

    # Phase 4: Final local search polish
    if time.time() < deadline - 1.0:
        final_perm, final_ms = _local_search(
            best_perm, processing_times, deadline - 0.5,
            use_insertion=True, use_swap=True, max_iterations=5000
        )
        if final_ms < best_ms:
            best_perm = final_perm
            best_ms = final_ms

    return list(best_perm)
# EVOLVE-BLOCK-END
