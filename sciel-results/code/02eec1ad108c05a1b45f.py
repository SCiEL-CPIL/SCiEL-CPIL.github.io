"""Rewritten program: weighted-majority-vote combinatorial optimizer for
low-discrepancy signs over all arithmetic progressions in [1,256]."""
import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin vs 90s wall-clock

    N = 256

    # ---- Build every arithmetic progression (start, positive step, len>=2) ----
    aps = []
    for d in range(1, N // 2 + 1):
        for start in range(0, N - d):
            indices = list(range(start, N, d))
            if len(indices) >= 2:
                aps.append(indices)
    num_aps = len(aps)

    # AP membership matrix (num_aps x N), Fortran order for contiguous columns
    M = np.zeros((num_aps, N), dtype=np.float32, order='F')
    for i, ap in enumerate(aps):
        for idx in ap:
            M[i, idx] = 1.0
    Mf = M

    # For fast local search: for each position, which APs contain it
    pos_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            pos_aps[idx].append(i)

    def evaluate(signs):
        sums = Mf.dot(signs.astype(np.float32))
        return int(np.max(np.abs(sums)))

    def vote(weights, signs):
        """Each sign = sign of weighted sum of APs containing that position."""
        contrib = weights.astype(np.float32) @ Mf  # (N,)
        s = np.sign(contrib)
        s[s == 0] = signs  # tie-break: keep current
        return s.astype(np.float32)

    rng = random.Random(7)

    best_signs = None
    best_cost = float('inf')

    def update_best(s, cost):
        nonlocal best_signs, best_cost
        if cost < best_cost:
            best_cost = cost
            best_signs = s.copy()

    # ---------------- Phase A: weighted-majority-vote iterations ----------------
    # Initial random signs
    signs = np.array([rng.choice([-1.0, 1.0]) for _ in range(N)], dtype=np.float32)

    # weights per AP: start with bias toward high-discrepancy APs
    sums = Mf.dot(signs)
    abs_sums = np.abs(sums)
    weights = np.maximum(abs_sums, 1.0).astype(np.float64)

    for it in range(2000):
        if time.time() >= deadline - 20.0:
            break
        new_signs = vote(weights, signs)
        cost = evaluate(new_signs)
        update_best(new_signs, cost)

        # refresh weights: emphasize currently-worst APs (geometric, decaying)
        sums = Mf.dot(new_signs)
        abs_sums = np.abs(sums)
        # decay old weights, add fresh emphasis on worst APs
        weights = 0.6 * weights + np.maximum(abs_sums, 1.0)
        signs = new_signs

    if best_signs is None:
        best_signs = signs.copy()
        best_cost = evaluate(best_signs)

    # ---------------- Phase B: fast local search (single flips) ----------------
    signs = best_signs.copy()
    cur = best_cost
    sums = Mf.dot(signs)  # current AP sums

    def try_flip(idx):
        nonlocal cur
        old = signs[idx]
        signs[idx] = -old
        delta = -2.0 * old
        # update only APs containing idx
        for ai in pos_aps[idx]:
            sums[ai] += delta
        new_cost = int(np.max(np.abs(sums)))
        if new_cost < cur:
            cur = new_cost
            return True
        else:
            signs[idx] = old
            for ai in pos_aps[idx]:
                sums[ai] -= delta
            return False

    # hill-climb to a local minimum
    improved = True
    while improved and time.time() < deadline - 8.0:
        improved = False
        for idx in range(N):
            if try_flip(idx):
                improved = True

    if cur < best_cost:
        best_cost = cur
        best_signs = signs.copy()

    # ---------------- Phase C: multi-flip simulated annealing ----------------
    signs = best_signs.copy()
    cur = best_cost
    sums = Mf.dot(signs)
    T = 3.0
    cool = 0.9997

    while time.time() < deadline - 1.0:
        if T < 0.05:
            break
        # flip a small random block (1-3 consecutive positions)
        k = rng.randrange(1, 4)
        start = rng.randrange(N - k + 1)
        idxs = list(range(start, start + k))

        old_vals = signs[idxs].copy()
        for idx in idxs:
            signs[idx] = -signs[idx]
        for idx in idxs:
            delta = -2.0 * old_vals[idxs.index(idx)]
            for ai in pos_aps[idx]:
                sums[ai] += delta
        new_cost = int(np.max(np.abs(sums)))

        if new_cost <= cur:
            cur = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            if rng.random() < np.exp(-(new_cost - cur) / T):
                cur = new_cost
            else:
                # revert
                for idx in idxs:
                    signs[idx] = old_vals[idxs.index(idx)]
                    delta = -2.0 * old_vals[idxs.index(idx)]
                    for ai in pos_aps[idx]:
                        sums[ai] -= delta
        T *= cool

    # final exact evaluation
    final_cost = evaluate(best_signs)
    if final_cost < best_cost:
        best_cost = final_cost

    return [int(x) for x in best_signs]
