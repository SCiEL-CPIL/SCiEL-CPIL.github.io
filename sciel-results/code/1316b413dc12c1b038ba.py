"""Ternary covering code K_3(7,1) with greedy + simulated annealing refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random


def _build_candidates(q, n):
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        cands = [h + list(tail)
                 for h in tetracode
                 for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        cands = [list(w) for w in itertools.product(range(q), repeat=n)]
    return cands


def _coverage_matrix(cands, all_words, radius):
    cands = np.asarray(cands, dtype=np.int8)
    all_words = np.asarray(all_words, dtype=np.int8)
    M = len(cands)
    N = len(all_words)
    cov = np.zeros((M, N), dtype=bool)
    chunk = 128
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        diff = cands[s:e, None, :].astype(np.int8) != all_words[None, :, :].astype(np.int8)
        dists = np.sum(diff, axis=2)
        cov[s:e, :] = dists <= radius
    return cov


def _greedy(cov, N, deadline):
    remaining = np.ones(N, dtype=np.int8)
    rem_count = N
    selected = []
    while rem_count > 0:
        if time.time() > deadline:
            break
        counts = cov.dot(remaining)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        selected.append(best)
        remaining[cov[best]] = 0
        rem_count = int(np.count_nonzero(remaining))
    return selected


def _remove_redundant(cov, selected, deadline):
    K = len(selected)
    if K == 0:
        return selected
    sel_cov = cov[np.array(selected), :]
    counts = np.sum(sel_cov, axis=0)
    removed = [False] * K
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        to_remove = []
        for i in range(K):
            if removed[i]:
                continue
            if not np.any(sel_cov[i] & (counts == 1)):
                to_remove.append(i)
        if to_remove:
            for i in to_remove:
                removed[i] = True
                counts -= sel_cov[i]
            changed = True
    return [selected[i] for i in range(K) if not removed[i]]


def _sa_refine(cov, selected, N, deadline, rng):
    M = cov.shape[0]
    selected = list(selected)
    K = len(selected)
    if K == 0:
        return selected

    counts = np.zeros(N, dtype=np.int32)
    in_set = np.zeros(M, dtype=bool)
    for i in selected:
        counts += cov[i]
        in_set[i] = True
    uncovered = int(np.count_nonzero(counts == 0))
    best_selected = list(selected)
    best_uncovered = uncovered
    best_K = K

    T = 2.0
    cool = 0.999
    t = T
    steps = 0
    max_steps = 300000
    while t > 0.01 and steps < max_steps:
        if time.time() > deadline:
            break
        steps += 1
        r = rng.random()
        if r < 0.5 and K > 1:
            move = "swap"
        elif r < 0.8 and K < M:
            move = "insert"
        else:
            move = "delete"

        do = False
        delta_unc = 0
        delta_K = 0
        idx = 0
        out_i = -1
        in_i = -1

        if move == "swap" and K > 1:
            idx = rng.randrange(K)
            out_i = selected[idx]
            if uncovered > 0:
                gains = cov.dot((counts == 0))
                gains[in_set] = -1
                in_i = int(np.argmax(gains))
                if gains[in_i] <= 0:
                    in_i = rng.randrange(M)
            else:
                in_i = rng.randrange(M)
            if in_i == out_i or in_set[in_i]:
                continue
            lose = int(np.count_nonzero(cov[out_i] & (counts == 1)))
            gain = int(np.count_nonzero(cov[in_i] & (counts == 0)))
            delta_unc = lose - gain
            delta_K = 0
            do = True
        elif move == "insert" and K < M:
            if uncovered > 0:
                gains = cov.dot((counts == 0))
                gains[in_set] = -1
                in_i = int(np.argmax(gains))
                if gains[in_i] <= 0:
                    in_i = rng.randrange(M)
            else:
                in_i = rng.randrange(M)
            if in_set[in_i]:
                continue
            gain = int(np.count_nonzero(cov[in_i] & (counts == 0)))
            delta_unc = -gain
            delta_K = 1
            do = True
        else:
            if K <= 1:
                continue
            idx = rng.randrange(K)
            out_i = selected[idx]
            lose = int(np.count_nonzero(cov[out_i] & (counts == 1)))
            delta_unc = lose
            delta_K = -1
            do = True

        if not do:
            continue

        if delta_unc < 0 or (delta_unc == 0 and delta_K < 0):
            accept = True
        else:
            cost = delta_unc + 0.5 * delta_K
            if cost <= 0:
                accept = True
            else:
                accept = (rng.random() < np.exp(-cost / t))

        if accept:
            if move == "swap":
                counts -= cov[out_i]
                counts += cov[in_i]
                in_set[out_i] = False
                in_set[in_i] = True
                selected[idx] = in_i
            elif move == "insert":
                counts += cov[in_i]
                in_set[in_i] = True
                selected.append(in_i)
                K += 1
            else:
                counts -= cov[out_i]
                in_set[out_i] = False
                selected.pop(idx)
                K -= 1

            unc_now = int(np.count_nonzero(counts == 0))
            if unc_now < best_uncovered or (unc_now == best_uncovered and K < best_K):
                best_uncovered = unc_now
                best_K = K
                best_selected = list(selected)

        t *= cool

    return best_selected


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    cands = _build_candidates(q, n)
    M = len(cands)
    cov = _coverage_matrix(cands, all_words, radius)

    selected = _greedy(cov, N, deadline - 70.0)
    selected = _remove_redundant(cov, selected, deadline - 65.0)

    rng = random.Random(12345)
    if time.time() < deadline - 10.0:
        selected = _sa_refine(cov, selected, N, deadline - 10.0, rng)

    selected = _remove_redundant(cov, selected, deadline - 5.0)

    if time.time() < deadline - 3.0:
        counts = np.sum(cov[np.array(selected)], axis=0)
        if np.any(counts == 0):
            selected = _greedy(cov, N, deadline - 2.0)
            selected = _remove_redundant(cov, selected, deadline - 1.0)

    return [cands[i].tolist() for i in selected]
# EVOLVE-BLOCK-END
