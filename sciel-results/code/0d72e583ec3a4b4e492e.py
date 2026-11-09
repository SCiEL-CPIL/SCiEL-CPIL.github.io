"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order. O(n_jobs * n_machines)."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    # C[j] = completion time of current job on machine j
    prev = np.zeros(n_machines)
    makespan = 0.0
    for job_idx in order:
        cur = np.zeros(n_machines)
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                cur[j] = prev[j] + p
            else:
                cur[j] = max(prev[j], cur[j - 1]) + p
        prev = cur
        makespan = prev[-1]
    return makespan


def _insertion_polish(processing_times, order, max_passes=2):
    """Insertion-based local search: try moving each job to every other position.
    Returns (improved_order, best_makespan)."""
    n = len(order)
    if n <= 1:
        return list(order), _makespan(processing_times, order)
    order = list(order)
    best_ms = _makespan(processing_times, order)
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            job = order[i]
            for j in range(n):
                if i == j:
                    continue
                # Move job from i to j
                new_order = order[:i] + order[i + 1:j + 1] + [job] + order[j + 1:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    order = new_order
                    i = j  # re-evaluate from new position
                    improved = True
                    break
        if not improved:
            break
    return order, best_ms


def _neh_rule(processing_times):
    """NEH heuristic: build permutation by inserting job with largest total
    processing time into the best position (by makespan)."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order_indices = np.argsort(-totals)
    order = [int(order_indices[0])]
    for idx in order_indices[1:]:
        idx = int(idx)
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [idx] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [idx] + order[best_pos:]
    return order


def order_jobs(processing_times):
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(42)
    n_jobs = processing_times.shape[0]

    # Strong initial solution via NEH
    best_order = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_order)
    best_order, best_ms = _insertion_polish(processing_times, best_order, max_passes=3)

    # Multi-restart simulated annealing with insertion moves
    def sa_run(start_order, start_ms, t_start):
        cur_order = list(start_order)
        cur_ms = start_ms
        local_best_order = list(cur_order)
        local_best_ms = cur_ms
        T = 1.0
        cool = 0.995
        n = len(cur_order)
        steps = 0
        while time.time() < t_start:
            # Random insertion move
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            new_order = cur_order[:i] + cur_order[i + 1:j + 1] + [cur_order[i]] + cur_order[j + 1:]
            new_ms = _makespan(processing_times, new_order)
            delta = new_ms - cur_ms
            if delta < 0 or rng.random() < np.exp(-delta / T):
                cur_order = new_order
                cur_ms = new_ms
                if cur_ms < local_best_ms:
                    local_best_ms = cur_ms
                    local_best_order = list(cur_order)
            T *= cool
            steps += 1
            # Periodic polish to escape local optima
            if steps % 2000 == 0:
                pol_order, pol_ms = _insertion_polish(processing_times, cur_order, max_passes=1)
                if pol_ms < local_best_ms:
                    local_best_ms = pol_ms
                    local_best_order = list(pol_order)
                    cur_order = list(pol_order)
                    cur_ms = pol_ms
        return local_best_order, local_best_ms

    # Run multiple SA restarts, each seeded from best with perturbation
    restarts = 0
    while time.time() < deadline:
        # Allocate a chunk of time for this restart
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        chunk = min(remaining - 1.0, 12.0)
        t_start = time.time() + chunk
        # Perturb the best solution for diversity
        cur = list(best_order)
        for _ in range(rng.randint(2, 5)):
            a = rng.randrange(len(cur))
            b = rng.randrange(len(cur))
            cur[a], cur[b] = cur[b], cur[a]
        cand_order, cand_ms = sa_run(cur, _makespan(processing_times, cur), t_start)
        if cand_ms < best_ms:
            best_ms = cand_ms
            best_order = list(cand_order)
        restarts += 1
        # Final polish on best
        best_order, best_ms = _insertion_polish(processing_times, best_order, max_passes=2)

    return list(best_order)
# EVOLVE-BLOCK-END
