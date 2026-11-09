"""Combined program: NEH/CDS variants + Tabu Search + VNS polish."""
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

def _insertion_cost(P, perm, job, pos):
    """Compute makespan of inserting `job` at position `pos` in `perm` (perm does not contain job)."""
    n = len(perm)
    m = P.shape[1]
    if m == 0:
        return 0.0
    if n == 0:
        return float(P[job].sum())
    prev_row = [0.0] * m
    curr_row = [0.0] * m
    for j in range(n + 1):
        if j < pos:
            job_j = perm[j]
        elif j == pos:
            job_j = job
        else:
            job_j = perm[j - 1]
        if j == 0:
            s = 0.0
            for k in range(m):
                s += P[job_j, k]
                curr_row[k] = s
        else:
            c0 = prev_row[0]
            curr_row[0] = c0 + P[job_j, 0]
            for k in range(1, m):
                cm1 = curr_row[k-1]
                pk = prev_row[k]
                curr_row[k] = (cm1 if cm1 > pk else pk) + P[job_j, k]
        prev_row, curr_row = curr_row, prev_row
    return prev_row[-1]

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
            ms = _insertion_cost(P, current_seq, job, pos)
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
            ms = _insertion_cost(P, current_seq, job, pos)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_seq.insert(best_pos, job)
    return current_seq

def _tabu_search(order, P, deadline=None, rng=None):
    """
    Tabu search with insertion moves and aspiration criterion.
    Tabu list stores (job, from_pos) to prevent immediate reversal.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    if rng is None:
        rng = np.random.RandomState(0)

    current_order = list(order)
    current_ms = _makespan(current_order, P)
    best_order = list(current_order)
    best_ms = current_ms

    # Tabu tenure: how many iterations a move stays tabu
    tabu_tenure = max(7, n_jobs // 3)
    # tabu_map: (job, from_pos) -> iteration at which it becomes non-tabu
    tabu_map = {}
    iteration = 0
    no_improve_count = 0
    max_no_improve = 200  # stop if no best improvement for this many iterations

    while True:
        if deadline is not None and time.time() > deadline:
            break

        best_move_i = -1
        best_move_j = -1
        best_move_ms = current_ms
        best_move_is_tabu = False

        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                ms = _insertion_cost(P, remaining, job, j)

                # Check tabu status
                move_key = (job, i)
                is_tabu = (move_key in tabu_map) and (tabu_map[move_key] > iteration)

                # Aspiration: accept tabu move if it beats global best
                if is_tabu and ms >= best_ms - 1e-12:
                    continue

                if ms < best_move_ms - 1e-12:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
                    best_move_is_tabu = is_tabu

        if best_move_i == -1:
            # No improving (or aspiration) move found; accept a random non-tabu move
            # or break if stuck
            # Try to find any non-tabu move
            any_move_i = -1
            any_move_j = -1
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = current_order[i]
                remaining = current_order[:i] + current_order[i+1:]
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    move_key = (job, i)
                    is_tabu = (move_key in tabu_map) and (tabu_map[move_key] > iteration)
                    if not is_tabu:
                        any_move_i = i
                        any_move_j = j
                        break
                if any_move_i >= 0:
                    break
            if any_move_i == -1:
                break
            best_move_i = any_move_i
            best_move_j = any_move_j
            best_move_ms = _insertion_cost(P,
                current_order[:best_move_i] + current_order[best_move_i+1:],
                current_order[best_move_i], best_move_j)

        # Apply the move
        job = current_order[best_move_i]
        remaining = current_order[:best_move_i] + current_order[best_move_i+1:]
        current_order = remaining[:best_move_j] + [job] + remaining[best_move_j:]
        current_ms = best_move_ms

        # Update tabu list: make the reverse move tabu
        # Reverse move: insert job at position best_move_i in the new permutation
        # In the new permutation, job is at position best_move_j
        # The reverse would be removing job from best_move_j and inserting at best_move_i
        # We tabu the move (job, best_move_j) for the next tabu_tenure iterations
        tabu_map[(job, best_move_j)] = iteration + tabu_tenure

        # Update best
        if current_ms < best_ms - 1e-12:
            best_ms = current_ms
            best_order = list(current_order)
            no_improve_count = 0
        else:
            no_improve_count += 1

        iteration += 1

        if no_improve_count > max_no_improve:
            break

        # Occasionally clear tabu list to allow exploration
        if iteration % 200 == 0:
            tabu_map.clear()

    return best_order

def _local_search_vns(order, P, deadline=None):
    """
    Variable Neighborhood Search:
      N1: adjacent swaps
      N2: 2-opt block reversals
      N3: Or-opt relocations
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan(current_order, P)

    max_cycles = 150

    for cycle in range(max_cycles):
        if deadline is not None and time.time() > deadline:
            break

        # N1: adjacent swaps - best-improvement
        best_swap_pos = -1
        best_swap_ms = current_ms
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(new_order, P)
            if new_ms < best_swap_ms - 1e-12:
                best_swap_ms = new_ms
                best_swap_pos = i
        if best_swap_pos >= 0:
            new_order = current_order[:]
            new_order[best_swap_pos], new_order[best_swap_pos+1] = new_order[best_swap_pos+1], new_order[best_swap_pos]
            current_order = new_order
            current_ms = best_swap_ms
            continue

        # N2: 2-opt reversals - best-improvement
        max_block = min(6, n_jobs)
        best_rev_start = -1
        best_rev_len = -1
        best_rev_ms = current_ms
        for block_len in range(2, max_block + 1):
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n_jobs - block_len + 1):
                if deadline is not None and time.time() > deadline:
                    break
                new_order = current_order[:]
                new_order[i:i+block_len] = reversed(new_order[i:i+block_len])
                new_ms = _makespan(new_order, P)
                if new_ms < best_rev_ms - 1e-12:
                    best_rev_ms = new_ms
                    best_rev_start = i
                    best_rev_len = block_len
        if best_rev_start >= 0:
            new_order = current_order[:]
            new_order[best_rev_start:best_rev_start+best_rev_len] = reversed(new_order[best_rev_start:best_rev_start+best_rev_len])
            current_order = new_order
            current_ms = best_rev_ms
            continue

        # N3: Or-opt relocations - best-improvement
        best_oropt = None
        best_oropt_ms = current_ms
        for block_size in [1, 2]:
            if n_jobs < block_size + 1:
                continue
            if deadline is not None and time.time() > deadline:
                break
            for i in range(n_jobs - block_size + 1):
                if deadline is not None and time.time() > deadline:
                    break
                block = current_order[i:i+block_size]
                remaining = current_order[:i] + current_order[i+block_size:]
                for j in range(len(remaining) + 1):
                    if deadline is not None and time.time() > deadline:
                        break
                    if j == i or (block_size == 1 and j == i + 1) or \
                       (block_size > 1 and j >= i and j < i + block_size):
                        continue
                    new_order = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan(new_order, P)
                    if new_ms < best_oropt_ms - 1e-12:
                        best_oropt_ms = new_ms
                        best_oropt = (i, j, block)
        if best_oropt is not None:
            i, j, block = best_oropt
            block_size = len(block)
            new_order = current_order[:i] + current_order[i+block_size:]
            new_order = new_order[:j] + block + new_order[j:]
            current_order = new_order
            current_ms = best_oropt_ms
            continue

        break

    return current_order

def _final_polish(order, P, deadline=None):
    """Final high-precision polish: exhaustive adjacent swaps and small relocations."""
    n_jobs = len(order)
    if n_jobs < 2:
        return list(order)

    current_order = list(order)
    current_ms = _makespan(current_order, P)

    for pass_num in range(10):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs - 1):
            if deadline is not None and time.time() > deadline:
                break
            new_order = current_order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms - 1e-15:
                current_order = new_order
                current_ms = new_ms
                improved = True
        if not improved:
            break

    for pass_num in range(5):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = current_order[i]
            remaining = current_order[:i] + current_order[i+1:]
            for j in range(len(remaining) + 1):
                if deadline is not None and time.time() > deadline:
                    break
                if j == i or j == i + 1:
                    continue
                new_order = remaining[:j] + [job] + remaining[j:]
                new_ms = _makespan(new_order, P)
                if new_ms < current_ms - 1e-15:
                    current_order = new_order
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
    Uses multiple NEH/CDS variants followed by Tabu Search and VNS polish.
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

    # Generate multiple initial solutions from different heuristics
    initial_perms = []

    # NEH variants
    for sort_key in ['total', 'first_last', 'first', 'last']:
        if time.time() > deadline - 20.0:
            break
        perm = _neh_heuristic(P, sort_key=sort_key)
        ms = _makespan(perm, P)
        initial_perms.append((ms, perm))
        if ms < best_ms:
            best_ms = ms
            best_perm = perm

    # CDS heuristic
    if time.time() < deadline - 20.0:
        perm_cds = _cds_heuristic(P)
        ms_cds = _makespan(perm_cds, P)
        initial_perms.append((ms_cds, perm_cds))
        if ms_cds < best_ms:
            best_ms = ms_cds
            best_perm = perm_cds

    # A few randomized NEH restarts
    rng = random.Random(12345)
    max_restarts = 6
    for r in range(max_restarts):
        if time.time() > deadline - 15.0:
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

    # Apply tabu search to the top initial solutions
    max_ls_attempts = min(3, len(initial_perms))
    ls_deadline = deadline - 5.0

    for idx in range(max_ls_attempts):
        if time.time() > ls_deadline:
            break
        ms_init, perm_init = initial_perms[idx]
        if ms_init > best_ms and idx > 0:
            continue

        improved_perm = _tabu_search(perm_init, P, deadline=ls_deadline)
        improved_ms = _makespan(improved_perm, P)

        if improved_ms < best_ms:
            best_ms = improved_ms
            best_perm = improved_perm

    # VNS polish on the best solution
    if time.time() < deadline - 2.0 and best_perm is not None:
        final_perm = _local_search_vns(best_perm, P, deadline=deadline - 1.0)
        final_ms = _makespan(final_perm, P)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm

    # Final high-precision polish
    if time.time() < deadline - 0.3 and best_perm is not None:
        polished_perm = _final_polish(best_perm, P, deadline=deadline - 0.3)
        polished_ms = _makespan(polished_perm, P)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm

    return list(best_perm)

# EVOLVE-BLOCK-END
