"""Combined program: fast NEH variants + insertion/Or-opt local search + block-reversal SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P (list-based, fast)."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0

    first_job = order[0]
    row = P[first_job]
    curr = [0.0] * n_machines
    curr[0] = row[0]
    for m in range(1, n_machines):
        curr[m] = curr[m - 1] + row[m]

    for idx in range(1, n_jobs):
        j = order[idx]
        row = P[j]
        new = [0.0] * n_machines
        new[0] = curr[0] + row[0]
        for m in range(1, n_machines):
            a = curr[m]
            b = new[m - 1]
            new[m] = (a if a > b else b) + row[m]
        curr = new
    return curr[-1]


def _neh(P, sort_key):
    """NEH heuristic with a given descending sort key."""
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    sorted_jobs = np.argsort(-sort_key)
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [int(j)] + perm[pos:]
            ms = _makespan(candidate, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, int(j))
    return perm


def _insertion_search(order, P, max_passes=30, start_time=None, time_limit=None):
    """Insertion local search: move each job to its best position."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)

    best_order = list(order)
    best_ms = _makespan(best_order, P)

    for _ in range(max_passes):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n):
            if time_limit is not None and time.time() - start_time > time_limit:
                return best_order, best_ms
            job = best_order[i]
            remaining = best_order[:i] + best_order[i + 1:]
            best_pos = i
            best_cand = best_ms
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(candidate, P)
                if ms < best_cand - 1e-10:
                    best_cand = ms
                    best_pos = pos
            if best_pos != i:
                remaining = best_order[:i] + best_order[i + 1:]
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_ms = best_cand
                improved = True
        if not improved:
            break
    return best_order, best_ms


def _or_opt_search(order, P, max_passes=10, start_time=None, time_limit=None):
    """Or-opt: move blocks of size 1-3 to other positions."""
    n = len(order)
    if n <= 3:
        return list(order), _makespan(order, P)

    best_order = list(order)
    best_ms = _makespan(best_order, P)

    for _ in range(max_passes):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        improved = False
        for block_size in [1, 2, 3]:
            if n < block_size + 1:
                continue
            for i in range(n - block_size + 1):
                if time_limit is not None and time.time() - start_time > time_limit:
                    return best_order, best_ms
                block = best_order[i:i + block_size]
                remaining = best_order[:i] + best_order[i + block_size:]
                for pos in range(len(remaining) + 1):
                    if pos == i or (block_size == 1 and pos == i + 1) or \
                       (block_size > 1 and pos >= i and pos < i + block_size):
                        continue
                    candidate = remaining[:pos] + block + remaining[pos:]
                    ms = _makespan(candidate, P)
                    if ms < best_ms - 1e-10:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        i = min(i, n - block_size)
                        break
        if not improved:
            break
    return best_order, best_ms


def _sa_block_reversal(order, P, start_time, time_limit, rng):
    """Simulated annealing with 2-opt block-reversal neighborhood."""
    n = len(order)
    if n <= 2:
        return list(order), _makespan(order, P)

    current = list(order)
    current_ms = _makespan(current, P)
    best = list(current)
    best_ms = current_ms

    initial_temp = max(current_ms * 0.05, 1.0)
    cooling = 0.995
    temp = initial_temp
    deadline = start_time + time_limit

    while temp > 1e-8 and time.time() < deadline:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        # Reverse block [i..j]
        new_order = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
        new_ms = _makespan(new_order, P)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current = new_order
            current_ms = new_ms
            if new_ms < best_ms:
                best = list(new_order)
                best_ms = new_ms
        temp *= cooling
    return best, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    P = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = P.shape

    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)

    # Step 1: diverse NEH starts
    candidates = []
    candidates.append(_neh(P, P.sum(axis=1)))
    candidates.append(_neh(P, P[:, 0]))
    candidates.append(_neh(P, P[:, n_machines - 1]))

    # Step 2: refine each with insertion search, keep best
    best_order, best_ms = None, float('inf')
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        polished, ms = _insertion_search(cand, P, max_passes=8,
                                         start_time=start_time, time_limit=deadline - start_time - 10.0)
        if ms < best_ms:
            best_ms = ms
            best_order = polished

    # Step 3: Or-opt polish
    if time.time() < deadline - 5.0:
        best_order, best_ms = _or_opt_search(best_order, P, max_passes=6,
                                             start_time=start_time, time_limit=deadline - start_time - 5.0)

    # Step 4: SA with block-reversal to escape local optima
    remaining = deadline - time.time() - 3.0
    if remaining > 3.0:
        sa_order, sa_ms = _sa_block_reversal(best_order, P, start_time, remaining, rng)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order

    # Step 5: final high-precision insertion polish
    remaining = deadline - time.time() - 1.0
    if remaining > 1.0:
        best_order, best_ms = _insertion_search(best_order, P, max_passes=15,
                                                start_time=start_time, time_limit=remaining)

    return list(best_order)
# EVOLVE-BLOCK-END
