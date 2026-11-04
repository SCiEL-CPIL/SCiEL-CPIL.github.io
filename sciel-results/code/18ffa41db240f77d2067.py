"""Covering design C(19,8,5): greedy set-cover + 2-opt local search.
Keeps construct(v,k,t) signature; returns list of k-subsets (lists of ints)."""
import time
import itertools
import numpy as np


def _binom_table(max_n, max_r):
    C = np.zeros((max_n + 1, max_r + 1), dtype=np.int32)
    C[:, 0] = 1
    for n in range(1, max_n + 1):
        C[n, 0] = 1
        for r in range(1, min(n, max_r) + 1):
            C[n, r] = C[n - 1, r - 1] + C[n - 1, r]
    return C


def _build_candidates(v, k):
    cands = [set(c) for c in itertools.combinations(range(v), k)]
    return cands


def _coverage_matrix(cands, v, k, t, binom):
    """M[i, j] = C(|cand_i & subset_j|, t)."""
    n = len(cands)
    m = int(binom[v, t])
    M = np.zeros((n, m), dtype=np.int32)
    for i, c in enumerate(cands):
        for x in c:
            idx = (x, k - x)
            M[i] += binom[idx[0], idx[1]]
    return M


def _subset_index(v, t, binom):
    """Map a t-subset (sorted tuple) to its index = sum_{x in s} C(v-1-x, t-1-x)."""
    def idx(s):
        total = 0
        for x in s:
            total += int(binom[v - 1 - x, t - 1 - x])
        return total
    return idx


def _greedy(cands, M, m, deadline):
    """Greedy: repeatedly add the block covering the most uncovered subsets."""
    n = len(cands)
    cov = np.zeros(m, dtype=np.int32)  # coverage count per subset
    chosen = []
    chosen_set = set()
    while True:
        if time.time() > deadline:
            break
        uncovered = (cov == 0).sum()
        if uncovered == 0:
            break
        # gain per candidate = number of subsets it covers that are currently uncovered
        gain = ((M > 0) & (cov[:, None] == 0)).sum(axis=1)  # n x m -> n
        # find best candidate not already chosen
        best = -1
        best_gain = 0
        for i in range(n):
            if i in chosen_set:
                continue
            g = gain[i]
            if g > best_gain:
                best_gain = g
                best = i
        if best == -1 or best_gain == 0:
            break
        chosen.append(best)
        chosen_set.add(best)
        cov += M[best]
    return chosen


def _coverage_bool(cands, v, t, binom, m):
    """C[i, j] = 1 if cand_i covers subset_j. Precomputed for local search."""
    n = len(cands)
    C = np.zeros((n, m), dtype=np.int8)
    for i, c in enumerate(cands):
        for x in c:
            idx = (x, k - x)
            C[i] += binom[idx[0], idx[1]]
    C = (C > 0).astype(np.int8)
    return C


def _local_search(chosen, cands, C, m, deadline):
    """2-opt: drop redundant blocks, swap blocks for better ones."""
    chosen = list(chosen)
    cov = np.zeros(m, dtype=np.int32)
    for i in chosen:
        cov += C[i]
    chosen_set = set(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # (a) drop redundant blocks
        for i in list(chosen):
            if cov[C[i] > 0].min() >= 2:  # every subset it covers is covered by >=2
                # safe to drop
                cov -= C[i]
                chosen.remove(i)
                chosen_set.discard(i)
                improved = True
        # (b) try swaps: for each chosen block, find a candidate (not chosen) with higher net gain
        for i in list(chosen):
            if time.time() > deadline:
                break
            # subsets that would become uncovered if i removed
            only_i = (cov == 1) & (C[i] > 0)
            n_only = only_i.sum()
            if n_only == 0:
                continue  # already redundant, handled above
            # best replacement candidate
            # net gain = (# subsets covered by cand) - (# subsets only covered by i and not by cand)
            # = (C[cand] > 0).sum() - (only_i & (C[cand]==0)).sum()
            cand_cov = (C > 0).sum(axis=1)  # precompute once per iteration is heavy; compute on fly
            # We'll compute gain vector: for each candidate j, gain_j = cand_cov[j] - (only_i & (C[j]==0)).sum()
            # only_i is a bool vector length m
            not_cover_only = (~C).astype(np.int8)  # n x m
            loss = (only_i[None, :] & not_cover_only).sum(axis=1)  # n
            gain_vec = cand_cov - loss
            gain_vec[list(chosen)] = -1
            j = int(np.argmax(gain_vec))
            if gain_vec[j] > 0 and j not in chosen_set:
                # perform swap
                cov -= C[i]
                cov += C[j]
                chosen.remove(i)
                chosen.append(j)
                chosen_set.discard(i)
                chosen_set.add(j)
                improved = True
    return chosen


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    binom = _binom_table(v, t)
    m = int(binom[v, t])

    cands = _build_candidates(v, k)
    n = len(cands)

    # Coverage count matrix
    M = _coverage_matrix(cands, v, k, t, binom)

    # Greedy
    chosen = _greedy(cands, M, m, deadline)

    if time.time() > deadline:
        return [sorted(cands[i]) for i in chosen]

    # Coverage bool matrix for local search
    C = _coverage_bool(cands, v, t, binom, m)

    # Local search
    chosen = _local_search(chosen, cands, C, m, deadline)

    return [sorted(cands[i]) for i in chosen]
