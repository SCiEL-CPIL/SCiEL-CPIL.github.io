"""Flow-shop permutation via NEH construction + time-budgeted local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 110.0  # wall-clock seconds; keep safety margin before the 150s kill


def _makespan(pt, order):
    """Makespan for a job order using the recurrence C[i,j]=max(C[i-1,j],C[i,j-1])+p[i,j]."""
    n = len(order)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # rows = jobs in order, cols = machines
    c = np.empty((n, m), dtype=np.float64)
    prev = pt[order[0]]
    c[0, :] = np.cumsum(prev)
    for i in range(1, n):
        cur = pt[order[i]]
        row = c[i, :]
        row[0] = c[i - 1, 0] + cur[0]
        if m > 1:
            row[1:] = np.maximum(c[i - 1, 1:], row[:-1]) + cur[1:]
    return c[n - 1, m - 1]


def _neh(pt):
    """NEH constructive heuristic: sort by total processing time, insert each job at the
    position minimizing the makespan of the partial schedule."""
    n = pt.shape[0]
    m = pt.shape[1]
    order = np.argsort(-pt.sum(axis=1))
    seq = [int(order[0])]
    c = np.empty((1, m), dtype=np.float64)
    c[0, :] = np.cumsum(pt[seq[0]])
    for k in range(1, n):
        job = int(order[k])
        pj = pt[job]
        best_c = None
        best_ms = np.inf
        for pos in range(k + 1):
            if pos == 0:
                nc0 = pj.copy()
                nc0[0] = c[0, 0] + pj[0]
                if m > 1:
                    nc0[1:] = np.maximum(c[0, 1:], nc0[:-1]) + pj[1:]
                ms = nc0[m - 1]
                if ms < best_ms:
                    best_ms = ms
                    best_c = nc0
            else:
                new = np.empty((k + 1, m), dtype=np.float64)
                new[:pos] = c[:pos]
                new[pos + 1:] = c[pos:]
                row = new[pos]
                row[0] = new[pos - 1, 0] + pj[0]
                if m > 1:
                    row[1:] = np.maximum(new[pos - 1, 1:], row[:-1]) + pj[1:]
                ms = row[m - 1]
                if ms < best_ms:
                    best_ms = ms
                    best_c = new
        seq.insert(pos, job)
        c = best_c
    return seq


def _local_search(pt, seq, deadline):
    """Or-opt style local search (move + swap) with first-improvement, time-budgeted."""
    n = len(seq)
    m = pt.shape[1]
    if n < 2:
        return seq
    best_ms = _makespan(pt, seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # move a job to another position
        for i in range(n):
            if time.time() >= deadline:
                return seq
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                seq2 = seq[:i] + seq[i + 1:j] + [seq[i]] + seq[j:]
                ms = _makespan(pt, seq2)
                if ms < best_ms - 1e-9:
                    seq = seq2
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # swap two adjacent-or-far jobs
        for i in range(n):
            if time.time() >= deadline:
                return seq
            for j in range(i + 1, n):
                seq2 = seq[:]
                seq2[i], seq2[j] = seq[j], seq[i]
                ms = _makespan(pt, seq2)
                if ms < best_ms - 1e-9:
                    seq = seq2
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    return seq


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    t0 = time.time()
    deadline = t0 + _DEADLINE
    n = pt.shape[0]
    if n <= 1:
        return list(range(n))
    seq = _neh(pt)
    seq = _local_search(pt, seq, deadline)
    return list(seq)
# EVOLVE-BLOCK-END
