"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """Compute makespan for a given job order. O(n*m)."""
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if len(order) == 0:
        return 0.0
    # completion[m] = completion time of the last processed job on machine m
    completion = np.zeros(n_machines)
    for j in order:
        row = processing_times[j]
        prev = completion[0]
        for m in range(n_machines):
            prev = prev + row[m]
            completion[m] = prev
    return completion[-1]


def _makespan_vec(order, processing_times):
    """Vectorized makespan using a running max recurrence. Faster for large n."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    P = processing_times[order]  # (n, m)
    # completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + P[j, m]
    # We can compute row by row, but vectorize across machines within a row.
    # C[:, 0] = cumsum(P[:, 0])
    # For m>=1: C[j, m] = C[j-1, m] + max(0, C[j, m-1] - C[j-1, m]) + P[j, m]
    n = P.shape[0]
    C = np.empty((n, n_machines))
    C[:, 0] = np.cumsum(P[:, 0])
    for m in range(1, n_machines):
        # C[j, m] = C[j-1, m] + max(0, C[j, m-1] - C[j-1, m]) + P[j, m]
        # For j=0: C[0, m] = C[0, m-1] + P[0, m]
        C[0, m] = C[0, m - 1] + P[0, m]
        if n > 1:
            prev_m = C[1:, m - 1]
            prev_m_same = C[:-1, m]
            delta = np.maximum(0.0, prev_m - prev_m_same)
            C[1:, m] = prev_m_same + delta + P[1:, m]
    return C[-1, -1]


def _neh(processing_times, rng=None):
    """NEH heuristic: insert jobs in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order_indices = np.argsort(-totals)
    # Build sequence by insertion
    seq = []
    for j in order_indices:
        best_pos = 0
        best_ms = np.inf
        # Try all insertion positions
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(cand, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def _swap_improve(seq, processing_times, max_iter=200):
    """Iterated swap local search. Returns improved seq and makespan."""
    n = len(seq)
    if n <= 1:
        return seq, _makespan(seq, processing_times)
    cur = np.array(seq)
    cur_ms = _makespan(cur, processing_times)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # swap i and j
                cur[i], cur[j] = cur[j], cur[i]
                ms = _makespan(cur, processing_times)
                if ms < cur_ms - 1e-9:
                    cur_ms = ms
                    improved = True
                else:
                    cur[i], cur[j] = cur[j], cur[i]
    return cur.tolist(), cur_ms


def _or_improve(seq, processing_times, max_iter=100):
    """Or-opt: move a block of 1-3 jobs to another position."""
    n = len(seq)
    if n <= 2:
        return seq, _makespan(seq, processing_times)
    cur = list(seq)
    cur_ms = _makespan(cur, processing_times)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for block_len in (1, 2, 3):
            for i in range(n - block_len + 1):
                block = cur[i:i + block_len]
                # remove block
                rest = cur[:i] + cur[i + block_len:]
                for pos in range(len(rest) + 1):
                    if pos == i:
                        continue
                    cand = rest[:pos] + block + rest[pos:]
                    ms = _makespan(cand, processing_times)
                    if ms < cur_ms - 1e-9:
                        cur = cand
                        cur_ms = ms
                        improved = True
                        # restart block search from top for this block_len
                        break
                if improved:
                    break
            if improved:
                break
    return cur, cur_ms


def _random_seq(n, rng):
    return rng.permutation(n).tolist()


def _perturb(seq, processing_times, rng, strength=3):
    """Randomly swap/insert a few jobs to escape local minima."""
    n = len(seq)
    cur = list(seq)
    for _ in range(strength):
        op = rng.random()
        if op < 0.5:
            # swap
            i, j = rng.sample(range(n), 2)
            cur[i], cur[j] = cur[j], cur[i]
        else:
            # or-opt move
            block_len = rng.choice([1, 2, 3])
            if n < block_len:
                continue
            i = rng.randrange(n - block_len + 1)
            block = cur[i:i + block_len]
            rest = cur[:i] + cur[i + block_len:]
            pos = rng.randrange(len(rest) + 1)
            if pos > i:
                pos -= 0  # index in rest
            cur = rest[:pos] + block + rest[pos:]
    return cur


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_machines == 0:
        return list(range(n_jobs))
    if n_jobs == 1:
        return [0]

    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    # Initial solution: NEH
    best_seq = _neh(processing_times, rng)
    best_ms = _makespan(best_seq, processing_times)

    # Local search on best
    best_seq, best_ms = _or_improve(best_seq, processing_times)
    best_seq, best_ms = _swap_improve(best_seq, processing_times)

    # Iterated local search with restarts
    # Each iteration: perturb best, local search, accept if better
    # Budget: keep running while time allows
    while time.time() < deadline:
        cand = _perturb(best_seq, processing_times, rng, strength=2)
        # local search
        cand, cand_ms = _or_improve(cand, processing_times)
        cand, cand_ms = _swap_improve(cand, processing_times, max_iter=50)
        if cand_ms < best_ms - 1e-9:
            best_seq = cand
            best_ms = cand_ms
        # occasionally try a fresh NEH variant or random start
        if rng.random() < 0.05 and time.time() < deadline - 10:
            alt = _random_seq(n_jobs, rng)
            alt, alt_ms = _or_improve(alt, processing_times)
            alt, alt_ms = _swap_improve(alt, processing_times, max_iter=30)
            if alt_ms < best_ms - 1e-9:
                best_seq = alt
                best_ms = alt_ms

    return list(best_seq)
# EVOLVE-BLOCK-END
