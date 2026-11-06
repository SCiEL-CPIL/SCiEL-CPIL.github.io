"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps

    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()

    return prefix_comps


def _compute_suffix_completions(perm, processing_times):
    """
    Compute backward completion vectors for every suffix.
    suffix_comps[i] represents the "remaining work" contribution of perm[i..n-1]
    such that the total makespan = max over m of (prefix_comps[i-1][m] + suffix_comps[i][m])
    (with prefix_comps[-1] = zeros).
    
    The recurrence: if we have jobs i..n-1 to process after the state D (completion
    times from the prefix), the final completion vector is computed forward.
    We want a function S_i such that for any input state D, the final state after
    processing jobs i..n-1 starting from D equals D + S_i (element-wise) when the
    dependency structure allows it. This is NOT generally true for flow-shop, but
    we can use the following: the makespan of the full schedule equals
    max_m (C_prefix[m] + S_i[m]) where S_i is computed by running the suffix
    with zero initial state and tracking the "incremental" contribution.
    
    Actually, for flow-shop, the correct decomposition is:
    If we define the suffix by processing jobs i..n-1 starting from zero initial
    completion times, we get some vector S_i. But the actual completion times
    when starting from D are NOT D + S_i in general.
    
    However, there's a well-known result: the makespan of the full permutation
    equals max over all "paths" in the job-machine DAG. We can decompose it as:
    makespan = max_m (C_prefix[m] + R_i[m])
    where R_i[m] is the minimum additional time needed on machine m and subsequent
    machines, given that machine m finishes the prefix at time C_prefix[m].
    
    A simpler correct approach: compute the "tail" vector T_i where T_i[m] is the
    makespan contribution of the suffix starting from machine m with zero initial
    state on all machines, but accounting for the fact that machine m starts at 0
    and subsequent machines must wait. This gives:
    T_n = zeros
    T_i[m] = processing_times[perm[i]][m] + max(T_{i+1}[m], T_i[m-1]) for m>0
    T_i[0] = processing_times[perm[i]][0] + T_{i+1}[0]
    
    Then the total makespan when prefix ends at C_prefix is:
    makespan = max_m (C_prefix[m] + T_i[m])
    
    Wait, this isn't quite right either. Let me think more carefully.
    
    The correct backward recurrence for flow-shop:
    Define B_i[m] as the "backward completion time" contribution.
    B_n[m] = 0 for all m.
    B_i[m] = p_i[m] + max(B_{i+1}[m], B_i[m-1])  for m >= 1
    B_i[0] = p_i[0] + B_{i+1}[0]
    
    Then the makespan of the full schedule = max_m (C_prefix[m] + B_i[m])
    where C_prefix is the completion vector after the prefix (jobs 0..i-1).
    
    Let me verify: for the last job (i = n-1), B_{n-1}[m] = p_{n-1}[m] + max(0, B_{n-1}[m-1]).
    The actual completion of the last job on machine m is max(C_prefix[m], C_{n-1}[m-1]) + p_{n-1}[m].
    And C_{n-1}[m-1] = C_prefix[m-1] + B_{n-1}[m-1]... this is getting circular.
    
    Let me just use a direct approach: compute the makespan of the full permutation
    by running the forward recurrence on the segment between prefix and suffix.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return [np.zeros(n_machines)]

    # B[i][m] = backward completion contribution
    # B[n] = zeros
    # B[i][0] = p[perm[i]][0] + B[i+1][0]
    # B[i][m] = p[perm[i]][m] + max(B[i+1][m], B[i][m-1])
    
    suffix_comps = [None] * (n_jobs + 1)
    suffix_comps[n_jobs] = np.zeros(n_machines)
    
    for i in range(n_jobs - 1, -1, -1):
        job = perm[i]
        pt = processing_times[job]
        B_next = suffix_comps[i + 1]
        B_cur = np.zeros(n_machines)
        B_cur[0] = pt[0] + B_next[0]
        for m in range(1, n_machines):
            B_cur[m] = pt[m] + max(B_next[m], B_cur[m-1])
        suffix_comps[i] = B_cur
    
    return suffix_comps


def _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1].
    Uses prefix completion at i-1 and suffix backward vector at i+2.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)

    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]

    job_a = perm[i]
    job_b = perm[i+1]

    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]

    # Process job_b first, then job_a
    C_b = np.zeros(n_machines)
    C_b[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_b[m] = max(C_prev[m], C_b[m-1]) + pt_b[m]

    C_a = np.zeros(n_machines)
    C_a[0] = C_b[0] + pt_a[0]
    for m in range(1, n_machines):
        C_a[m] = max(C_b[m], C_a[m-1]) + pt_a[m]

    if i + 2 >= n_jobs:
        return float(np.max(C_a))
    else:
        B_suffix = suffix_comps[i+2]
        return float(np.max(C_a + B_suffix))


def _eval_insert_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)

    if i == j:
        return _makespan(perm, processing_times)

    if i < j:
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1]

        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new

        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_mid[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_mid[m], C_a[m-1]) + pt_a[m]

        if j + 1 >= n_jobs:
            return float(np.max(C_a))
        else:
            B_suffix = suffix_comps[j+1]
            return float(np.max(C_a + B_suffix))

    else:
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1]

        job_a = perm[i]
        pt_a = processing_times[job_a]
        C_a = np.zeros(n_machines)
        C_a[0] = C_prev[0] + pt_a[0]
        for m in range(1, n_machines):
            C_a[m] = max(C_prev[m], C_a[m-1]) + pt_a[m]

        C_mid = C_a
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            C_new = np.zeros(n_machines)
            C_new[0] = C_mid[0] + pt[0]
            for m in range(1, n_machines):
                C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
            C_mid = C_new

        if i + 1 >= n_jobs:
            return float(np.max(C_mid))
        else:
            B_suffix = suffix_comps[i+1]
            return float(np.max(C_mid + B_suffix))


def _eval_reversal_incremental(perm, i, j, processing_times, prefix_comps, suffix_comps):
    """
    Evaluate makespan of reversing segment perm[i..j] (2-opt move).
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)

    if j - i < 1:
        return _makespan(perm, processing_times)

    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1]

    # Process reversed segment: perm[j], perm[j-1], ..., perm[i]
    C_mid = C_prev
    for k in range(j, i-1, -1):
        job = perm[k]
        pt = processing_times[job]
        C_new = np.zeros(n_machines)
        C_new[0] = C_mid[0] + pt[0]
        for m in range(1, n_machines):
            C_new[m] = max(C_mid[m], C_new[m-1]) + pt[m]
        C_mid = C_new

    if j + 1 >= n_jobs:
        return float(np.max(C_mid))
    else:
        B_suffix = suffix_comps[j+1]
        return float(np.max(C_mid + B_suffix))


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')

    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos

    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()

    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)

    order = list(np.argsort(-totals))

    if n_jobs > 2:
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]

    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)

    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()

    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))

    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)

    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)

    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _reverse_segment(perm, i, j):
    """Reverse segment perm[i..j] inclusive."""
    new_perm = list(perm)
    new_perm[i:j+1] = reversed(new_perm[i:j+1])
    return new_perm


def _precompute_swap_deltas(perm, processing_times):
    """
    Precompute all adjacent swap deltas using prefix/suffix completions.
    Returns array of delta values for each adjacent swap position.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]

    if n_jobs <= 1:
        return np.array([])

    prefix_comps = _compute_prefix_completions(perm, processing_times)
    suffix_comps = _compute_suffix_completions(perm, processing_times)

    current_ms = _makespan(perm, processing_times)
    deltas = np.zeros(n_jobs - 1)

    for i in range(n_jobs - 1):
        ms_after = _eval_swap_incremental(perm, i, processing_times, prefix_comps, suffix_comps)
        deltas[i] = ms_after - current_ms

    return deltas


def _fast_local_search(perm, processing_times, max_iterations=500, deadline=None):
    """
    Fast local search using precomputed swap deltas for O(1) adjacent swap evaluation.
    Also includes insert moves and 2-opt reversal moves.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)

    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)

    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break

        improved = False

        # Try adjacent swaps first (cheapest)
        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)

        deltas = np.zeros(n_jobs - 1)
        for i in range(n_jobs - 1):
            ms_after = _eval_swap_incremental(best_perm, i, processing_times, prefix_comps, suffix_comps)
            deltas[i] = ms_after - best_ms

        min_idx = np.argmin(deltas)
        if deltas[min_idx] < -1e-10:
            best_perm[min_idx], best_perm[min_idx+1] = best_perm[min_idx+1], best_perm[min_idx]
            best_ms += deltas[min_idx]
            improved = True

        if not improved:
            # Try insert moves
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    break
                job = best_perm[i]
                for pos in range(n_jobs):
                    if pos == i or pos == i + 1:
                        continue
                    ms = _eval_insert_incremental(best_perm, i, pos, processing_times, prefix_comps, suffix_comps)
                    if ms < best_ms - 1e-10:
                        new_perm = _move_job(best_perm, job, pos)
                        best_perm = new_perm
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            # Try 2-opt reversal moves (only for small instances to avoid O(n^3))
            if n_jobs <= 40:
                for i in range(n_jobs - 1):
                    if deadline is not None and time.time() > deadline:
                        break
                    for j in range(i + 1, n_jobs):
                        if j - i < 2:  # skip adjacent (already handled)
                            continue
                        ms = _eval_reversal_incremental(best_perm, i, j, processing_times, prefix_comps, suffix_comps)
                        if ms < best_ms - 1e-10:
                            best_perm = _reverse_segment(best_perm, i, j)
                            best_ms = ms
                            improved = True
                            break
                    if improved:
                        break

        if not improved:
            break

    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with adjacent swap, insert, and reversal moves.
    """
    if rng is None:
        rng = random.Random()

    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)

    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms

    initial_temp = max(current_ms * 0.15, 1.0)
    cooling_rate = 0.998

    temp = initial_temp
    no_improve_count = 0

    while time.time() < deadline:
        if temp < 1e-10:
            break

        move_type = rng.random()
        if move_type < 0.4:
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        elif move_type < 0.8:
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        else:
            # Reversal move
            if n_jobs <= 50:
                i = rng.randint(0, n_jobs - 2)
                j = rng.randint(i + 1, n_jobs - 1)
                if j - i < 2:
                    continue
                new_perm = _reverse_segment(current_perm, i, j)
            else:
                i = rng.randint(0, n_jobs - 2)
                new_perm = _swap_adjacent(current_perm, i)

        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms

        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms

            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        temp *= cooling_rate

        # Adaptive: if stuck for too long, perturb
        if no_improve_count > 1000:
            # Random restart perturbation
            for _ in range(3):
                i = rng.randint(0, n_jobs - 2)
                current_perm = _swap_adjacent(current_perm, i)
            current_ms = _makespan(current_perm, processing_times)
            no_improve_count = 0
            temp = max(temp, initial_temp * 0.1)

    return best_perm


def _pair_swap_local_search(perm, processing_times, deadline=None):
    """
    Local search using pair swaps (any two positions) evaluated incrementally.
    """
    n_jobs = len(perm)
    if n_jobs <= 2:
        return list(perm)

    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)

    for it in range(200):
        if deadline is not None and time.time() > deadline:
            break

        prefix_comps = _compute_prefix_completions(best_perm, processing_times)
        suffix_comps = _compute_suffix_completions(best_perm, processing_times)

        improved = False
        # Try pair swaps in a random order for diversity
        positions = list(range(n_jobs))
        random.shuffle(positions)

        for i in positions:
            if deadline is not None and time.time() > deadline:
                break
            for j in positions:
                if j <= i:
                    continue
                if abs(i - j) == 1:
                    continue  # adjacent already handled
                # Evaluate: move job at i to j (or equivalently swap)
                # Use insert evaluation: remove i, insert at j
                ms = _eval_insert_incremental(best_perm, i, j, processing_times, prefix_comps, suffix_comps)
                if ms < best_ms - 1e-10:
                    best_perm = _move_job(best_perm, best_perm[i], j)
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit

    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    rng = random.Random(42)

    candidates = []

    # LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)

    # NEH variants
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)

    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)

    if time.time() < deadline:
        neh3_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh3_perm)

    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)

    # Random initializations
    for _ in range(3):
        if time.time() < deadline:
            rand_perm = list(range(n_jobs))
            rng.shuffle(rand_perm)
            candidates.append(rand_perm)

    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)

    # Phase 1: Fast local search (adjacent swaps + inserts + 2-opt)
    if time.time() < deadline and n_jobs > 1:
        if n_jobs <= 30:
            max_iter = 3000
        elif n_jobs <= 50:
            max_iter = 1500
        elif n_jobs <= 100:
            max_iter = 600
        else:
            max_iter = 250

        polished = _fast_local_search(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms

    # Phase 2: Pair swap local search
    if time.time() < deadline - 5.0 and n_jobs > 2:
        pair_polished = _pair_swap_local_search(best_perm, processing_times, deadline=deadline - 5.0)
        pair_ms = _makespan(pair_polished, processing_times)
        if pair_ms < best_ms:
            best_perm = pair_polished
            best_ms = pair_ms

    # Phase 3: Simulated annealing
    if time.time() < deadline - 10.0 and n_jobs > 1:
        sa_deadline = deadline - 10.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2.0:
        if n_jobs <= 30:
            max_iter_final = 2000
        elif n_jobs <= 50:
            max_iter_final = 1000
        else:
            max_iter_final = 400
        final_polish = _fast_local_search(best_perm, processing_times, max_iterations=max_iter_final, deadline=deadline - 2.0)
        final_ms = _makespan(final_polish, processing_times)
        if final_ms < best_ms:
            best_perm = final_polish
            best_ms = final_ms

    # Phase 5: Final pair swap polish
    if time.time() < deadline - 1.0 and n_jobs > 2:
        final_pair = _pair_swap_local_search(best_perm, processing_times, deadline=deadline - 1.0)
        final_pair_ms = _makespan(final_pair, processing_times)
        if final_pair_ms < best_ms:
            best_perm = final_pair
            best_ms = final_pair_ms

    return list(best_perm)
# EVOLVE-BLOCK-END
