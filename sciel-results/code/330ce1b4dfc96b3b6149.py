"""Randomized greedy + local-search covering for C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from scipy import sparse


def _all_blocks(v, k):
    return [itertools.combinations(range(v), k)]


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin vs 150s wall-clock

    # Enumerate all t-subsets (the "targets") and all k-subsets (the "blocks")
    tsubs = list(itertools.combinations(range(v), t))
    nT = len(tsubs)
    t_index = {ts: i for i, ts in enumerate(tsubs)}

    # All k-subsets
    ksubs = list(itertools.combinations(range(v), k))
    nK = len(ksubs)

    # Incidence: for each k-subset, list of t-subset indices it covers
    block_to_t = []
    for ks in ksubs:
        ks_set = set(ks)
        idxs = [t_index[ts] for ts in itertools.combinations(ks, t)]
        block_to_t.append(idxs)

    # Precompute, for each t-subset, the list of k-subsets covering it (for local search)
    t_to_k = [[] for _ in range(nT)]
    for bi, idxs in enumerate(block_to_t):
        for ti in idxs:
            t_to_k[ti].append(bi)

    # Sparse incidence matrix A: rows=t-subsets, cols=k-subsets, A[ti,bi]=1 if covered
    rows, cols = [], []
    for bi, idxs in enumerate(block_to_t):
        for ti in idxs:
            rows.append(ti)
            cols.append(bi)
    A = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (np.array(rows, dtype=np.int32),
         np.array(cols, dtype=np.int32))), shape=(nT, nK))

    def coverage_of(block_set):
        # returns array of coverage counts per t-subset
        if not block_set:
            return np.zeros(nT, dtype=np.int32)
        idx = np.array(sorted(block_set), dtype=np.int32)
        sub = A[:, idx]
        return sub.sum(axis=1).astype(np.int32).ravel()

    def is_covered(cov):
        return cov.min() >= 1

    # ---- Phase 1: randomized greedy (vectorized) ----
    def greedy_random(seed):
        rng = np.random.default_rng(seed)
        cov = np.zeros(nT, dtype=np.int32)
        chosen = []
        remaining = np.ones(nK, dtype=bool)
        while not is_covered(cov):
            if time.time() > DEADLINE - 5.0:
                break
            # new coverage if we add block b = A[:,b] where cov==0
            uncovered_mask = (cov == 0)
            if not uncovered_mask.any():
                break
            # gain[b] = number of uncovered t-subsets covered by b
            gain = A[uncovered_mask, :].sum(axis=0).ravel().astype(np.int64)
            # restrict to remaining
            gain = np.where(remaining, gain, -1)
            # add small random tie-break to diversify
            gain = gain + rng.random(nK) * 0.01
            b = int(np.argmax(gain))
            if gain[b] <= 0:
                break
            chosen.append(b)
            cov += A[:, b].toarray().ravel().astype(np.int32)
            remaining[b] = False
        return chosen, cov

    best_blocks = None
    best_cov = None
    best_n = None
    n_trials = 6
    for s in range(n_trials):
        if time.time() > DEADLINE - 30.0:
            break
        chosen, cov = greedy_random(s)
        if is_covered(cov):
            n = len(chosen)
            if best_n is None or n < best_n:
                best_n = n
                best_blocks = list(chosen)
                best_cov = cov

    if best_blocks is None:
        # Fallback: deterministic greedy (seed 0) should have worked; force one more
        chosen, cov = greedy_random(0)
        best_blocks = list(chosen)
        best_cov = cov
        best_n = len(chosen)

    # ---- Phase 2: local search polish (single-block replacement) ----
    # For each chosen block, try replacing it with another block such that
    # net coverage does not decrease (i.e., every t-subset still covered).
    chosen_set = set(best_blocks)
    chosen_arr = np.array(best_blocks, dtype=np.int32)
    cov = coverage_of(best_blocks)

    # candidate pool: all blocks not currently chosen
    not_chosen = [bi for bi in range(nK) if bi not in chosen_set]

    def try_polish():
        nonlocal cov, best_n, best_blocks
        improved = True
        while improved and time.time() < DEADLINE - 8.0:
            improved = False
            cur_n = len(best_blocks)
            for i, bi in enumerate(best_blocks):
                if time.time() > DEADLINE - 8.0:
                    return
                # coverage if we remove block bi
                cov_without = cov - A[:, bi].toarray().ravel().astype(np.int32)
                # t-subsets that would become uncovered
                newly_uncovered = np.where(cov_without == 0)[0]
                if len(newly_uncovered) == 0:
                    # removing bi alone keeps coverage -> we can drop it!
                    best_blocks.pop(i)
                    best_blocks = best_blocks[:i] + best_blocks[i+1:]
                    cov = coverage_of(best_blocks)
                    best_n = len(best_blocks)
                    improved = True
                    break
                # find a block (not bi, not in chosen) that covers all newly_uncovered
                # and whose addition keeps everything covered
                # a replacement b is valid if cov_without + A[:,b] >= 1 everywhere,
                # i.e., A[:,b] covers all newly_uncovered (since cov_without>=1 elsewhere)
                need = set(newly_uncovered.tolist())
                # quick filter: candidate must cover each needed t-subset
                cand = []
                for b in not_chosen:
                    if b == bi:
                        continue
                    # check coverage of needed t-subsets
                    sub = A[need, b].toarray().ravel()
                    if sub.min() >= 1:
                        cand.append(b)
                if not cand:
                    continue
                # pick the candidate that minimizes total blocks added? we add exactly 1,
                # so any works; but prefer one that is "redundant" (covers many already-covered)
                # to leave room. Just take first for now, but try a few.
                # Replace bi with best candidate (fewest extra overlaps is fine; all add 1)
                b_new = cand[0]
                best_blocks[i] = b_new
                cov = coverage_of(best_blocks)
                best_n = len(best_blocks)
                improved = True
                break
        return improved

    try_polish()

    # ---- Phase 3: second polish pass with fresh random restarts of local search ----
    # Re-run greedy a couple more times if time allows, then polish each.
    for s in range(n_trials, n_trials + 4):
        if time.time() > DEADLINE - 20.0:
            break
        chosen, cov2 = greedy_random(s)
        if not is_covered(cov2):
            continue
        if len(chosen) < best_n:
            best_blocks = list(chosen)
            best_n = len(chosen)
            cov = cov2
            try_polish()

    # ---- Final validation & conversion to point lists ----
    # Ensure coverage
    final_cov = coverage_of(best_blocks)
    if not is_covered(final_cov):
        # safety: fall back to a guaranteed greedy cover (seed 0, no time limit on coverage)
        cov = np.zeros(nT, dtype=np.int32)
        chosen = []
        remaining = np.ones(nK, dtype=bool)
        while not is_covered(cov):
            uncovered_mask = (cov == 0)
            gain = A[uncovered_mask, :].sum(axis=0).ravel().astype(np.int64)
            gain = np.where(remaining, gain, -1)
            b = int(np.argmax(gain))
            if gain[b] <= 0:
                break
            chosen.append(b)
            cov += A[:, b].toarray().ravel().astype(np.int32)
            remaining[b] = False
        best_blocks = chosen

    # Deduplicate (shouldn't be needed) and convert
    seen = set()
    out = []
    for bi in best_blocks:
        if bi not in seen:
            seen.add(bi)
            out.append(list(ksubs[bi]))
    # Sort each block for determinism (already sorted from combinations)
    return out
# EVOLVE-BLOCK-END
