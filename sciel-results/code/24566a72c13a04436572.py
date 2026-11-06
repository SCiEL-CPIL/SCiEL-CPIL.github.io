import time
import math
import random
import numpy as np


def _build_ap_tables(n):
    """Precompute all arithmetic progressions.

    Returns:
      starts: list of arrays, each the index array of one AP (length>=2).
      idx:    flat list of (array, index-in-array) for every element occurrence.
    """
    starts = []
    idx = []
    for d in range(1, n):
        for s in range(0, n - d + 1):
            arr = np.arange(s, n, d)
            if len(arr) < 2:
                continue
            starts.append(arr)
            base = len(idx)
            for j in range(len(arr)):
                idx.append((arr, j))
    return starts, idx


def _compute_cost(x, starts, idx):
    """Max over all APs of (max prefix sum - min prefix sum)."""
    best = 0
    for arr in starts:
        p = np.cumsum(x[arr])
        m = p.max() - p.min()
        if m > best:
            best = m
    return best


def _compute_greedy(n, starts, idx, time_limit):
    """Greedy sequential construction minimizing running max discrepancy."""
    x = np.zeros(n, dtype=np.int64)
    mins = np.zeros(len(starts), dtype=np.int64)
    maxs = np.zeros(len(starts), dtype=np.int64)
    best = 0
    start_time = time.time()
    for i in range(n):
        best_val = None
        best_cost = None
        for v in (-1, 1):
            x[i] = v
            c = 0
            for arr, j in idx:
                if arr[j] == i:
                    prev = maxs[arr[j - 1]]
                    new = prev + v
                    if new > maxs[arr[j]]:
                        maxs[arr[j]] = new
                    if new < mins[arr[j]]:
                        mins[arr[j]] = new
                    d = maxs[arr[j]] - mins[arr[j]]
                    if d > c:
                        c = d
            x[i] = 0
            if best_cost is None or c < best_cost:
                best_cost = c
                best_val = v
            if time.time() - start_time > time_limit:
                break
        x[i] = best_val if best_val is not None else 1
        for arr, j in idx:
            if arr[j] == i:
                prev = maxs[arr[j - 1]]
                new = prev + best_val
                if new > maxs[arr[j]]:
                    maxs[arr[j]] = new
                if new < mins[arr[j]]:
                    mins[arr[j]] = new
                d = maxs[arr[j]] - mins[arr[j]]
                if d > best:
                    best = d
    return x, best


def construct():
    n = 256
    start_time = time.time()
    time_limit = 78.0  # safety margin vs 90s wall-clock

    starts, idx = _build_ap_tables(n)

    # --- Greedy initialization for a strong starting point ---
    gtime = 18.0
    x, current_cost = _compute_greedy(n, starts, idx, gtime)
    if time.time() - start_time > time_limit:
        return [int(v) for v in x]

    mins = np.zeros(len(starts), dtype=np.int64)
    maxs = np.zeros(len(starts), dtype=np.int64)
    for arr in starts:
        p = np.cumsum(x[arr])
        mins[arr[0]] = p.min()
        maxs[arr[0]] = p.max()

    best_cost = current_cost
    best_x = x.copy()

    T = 3.0
    T_min = 0.02
    alpha = 0.9995
    r = random.Random(12345)

    while time.time() - start_time < time_limit:
        i = r.randrange(n)
        old = x[i]
        new = -old
        x[i] = new
        c = 0
        for arr, j in idx:
            if arr[j] == i:
                prev = maxs[arr[j - 1]]
                nv = prev + new
                if nv > maxs[arr[j]]:
                    maxs[arr[j]] = nv
                if nv < mins[arr[j]]:
                    mins[arr[j]] = nv
                d = maxs[arr[j]] - mins[arr[j]]
                if d > c:
                    c = d
        if c < current_cost:
            current_cost = c
            if c < best_cost:
                best_cost = c
                best_x = x.copy()
        else:
            delta = c - current_cost
            if delta > 0:
                if r.random() < math.exp(-delta / T):
                    current_cost = c
                else:
                    x[i] = old
                    for arr, j in idx:
                        if arr[j] == i:
                            prev = maxs[arr[j - 1]]
                            nv = prev + old
                            if nv > maxs[arr[j]]:
                                maxs[arr[j]] = nv
                            if nv < mins[arr[j]]:
                                mins[arr[j]] = nv
        T *= alpha
        if T < T_min:
            T = T_min

    return [int(v) for v in best_x]
