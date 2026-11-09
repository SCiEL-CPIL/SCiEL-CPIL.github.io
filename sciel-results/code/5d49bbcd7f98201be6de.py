"""Flow-shop permutation optimizer: fast makespan + multi-restart local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 1.5


def _makespan(pt, perm):
    """Exact makespan for a fixed job order, O(n*m) with a single pass."""
    n, m = pt.shape
    c = 0.0
    for j in perm:
        t = 0.0
        for k in range(m):
            t += pt[j, k]
            if t < c:
                t = c
        c = t
    return c


def _makespan_row(pt, perm):
    """Makespan using a row-major flattened machine vector per job."""
    n, m = pt.shape
    c = 0.0
    for j in perm:
        t = 0.0
        for v in pt[j]:
            t += v
            if t < c:
                t = c
        c = t
    return c


def _neh(pt, totals):
    """NEH heuristic: build by inserting job with largest total at best position."""
    order = list(np.argsort(-totals))
    seq = []
    for j in order:
        best_i, best_c = 0, None
        row = pt[j]
        for i in range(len(seq) + 1):
            cand = seq[:i] + [j] + seq[i:]
            c = _makespan_row(pt, cand)
            if best_c is None or c < best_c:
                best_c, best_i = c, i
        seq.insert(best_i, j)
    return seq


def _improve(pt, seq, ms, deadline):
    """2-opt + insertion local search; returns improved sequence and makespan."""
    seq = list(seq)
    n = len(seq)
    best = ms
    improved = True
    while improved:
        improved = False
        # 2-opt: reverse a segment
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                c = _makespan_row(pt, cand)
                if c < best - 1e-12:
                    best, seq, improved = c, cand, True
        if time.time() > deadline:
            break
        # insertion: move a job to a new position
        for i in range(n):
            if time.time() > deadline:
                break
            jv = seq[i]
            base = seq[:i] + seq[i + 1:]
            for k in range(len(base)):
                cand = base[:k] + [jv] + base[k:]
                c = _makespan_row(pt, cand)
                if c < best - 1e-12:
                    best, seq, improved = c, cand, True
    return seq, best


def order_jobs(processing_times):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    pt = np.asarray(processing_times, dtype=float)
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]

    totals = pt.sum(axis=1)

    # Candidate initial solutions from multiple heuristics + randomness.
    rng = np.random.default_rng(12345)
    starts = []
    starts.append(list(np.argsort(-totals)))            # LPT
    starts.append(_neh(pt, totals))                     # NEH
    for _ in range(3):
        starts.append(list(rng.permutation(n)))         # random
    starts.append(list(np.argsort(-pt[:, 0])))          # first-machine
    starts.append(list(np.argsort(-pt[:, -1])))         # last-machine

    best_seq = None
    best_ms = None
    seen = set()
    for s in starts:
        if time.time() > deadline:
            break
        key = tuple(s)
        if key in seen:
            continue
        seen.add(key)
        ms = _makespan_row(pt, s)
        if best_ms is None or ms < best_ms:
            best_ms, best_seq = ms, s
        # local search on each start (budget-limited)
        if time.time() < deadline:
            ns, nms = _improve(pt, s, ms, deadline)
            if nms < best_ms - 1e-12:
                best_ms, best_seq = nms, ns
            # refine the best-so-far again
            if time.time() < deadline:
                ns2, nms2 = _improve(pt, best_seq, best_ms, deadline)
                if nms2 < best_ms - 1e-12:
                    best_ms, best_seq = nms2, ns2

    if best_seq is None:
        best_seq = list(range(n))
    return list(best_seq)
# EVOLVE-BLOCK-END
