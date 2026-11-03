"""Covering design C(19,8,5): minimize 8-blocks covering every 5-subset."""
import time
import itertools
import numpy as np

_DEADLINE = 112.0  # safety margin under 120s wall-clock


def construct(v, k, t):
    t0 = time.time()

    # ---- Precompute all t-subsets and all k-blocks ---------------------
    t_subs = np.array(list(itertools.combinations(range(v), t)), dtype=np.int32)
    n_t = t_subs.shape[0]
    k_blocks = np.array(list(itertools.combinations(range(v), k)), dtype=np.int32)
    n_k = k_blocks.shape[0]

    # Map each t-subset (sorted tuple) -> index
    sub_index = {tuple(row): i for i, row in enumerate(t_subs)}

    # Incidence: block b covers its C(k,t) t-subsets
    pairs = []
    for b in range(n_k):
        blk = k_blocks[b]
        for s in itertools.combinations(blk, t):
            pairs.append((b, sub_index[s]))
    pairs = np.array(pairs, dtype=np.int32)
    inc = np.zeros((n_k, n_t), dtype=np.int8)
    inc[pairs[:, 0], pairs[:, 1]] = 1

    # ---- Greedy construction (vectorized) ------------------------------
    uncovered = np.ones(n_t, dtype=np.int8)
    in_use = np.zeros(n_k, dtype=np.int8)

    def remaining():
        return int(uncovered.sum())

    while remaining() > 0:
        cov = inc @ uncovered
        best = int(np.argmax(cov))
        if cov[best] == 0:
            break
        in_use[best] = 1
        uncovered[inc[best] == 1] = 0
        if time.time() - t0 > _DEADLINE - 25:
            break

    # Finish if greedy hit deadline with gaps
    while remaining() > 0 and time.time() - t0 < _DEADLINE - 3:
        cov = inc @ uncovered
        best = int(np.argmax(cov))
        if cov[best] == 0:
            break
        in_use[best] = 1
        uncovered[inc[best] == 1] = 0

    # ---- Local search to reduce block count ----------------------------
    def recompute():
        nonlocal uncovered
        covered = (inc.T @ in_use) > 0
        uncovered = (~covered).astype(np.int8)

    def local_search():
        improved = True
        while improved and time.time() - t0 < _DEADLINE - 8:
            improved = False
            recompute()
            in_use_idx = np.where(in_use == 1)[0]
            # 1) drop redundant blocks
            for bi in in_use_idx:
                if time.time() - t0 > _DEADLINE - 8:
                    break
                if in_use[bi] == 0:
                    continue
                in_use[bi] = 0
                recompute()
                if int(uncovered.sum()) == 0:
                    improved = True
                else:
                    in_use[bi] = 1
                    recompute()
            # 2) best 1-for-1 swap
            in_use_idx = np.where(in_use == 1)[0]
            for bi in in_use_idx:
                if time.time() - t0 > _DEADLINE - 8:
                    break
                if in_use[bi] == 0:
                    continue
                without = in_use.copy()
                without[bi] = 0
                cur = (inc.T @ without) > 0
                uniq = (~cur).astype(np.int8)
                need = int(uniq.sum())
                if need == 0:
                    in_use[bi] = 0
                    recompute()
                    improved = True
                    continue
                cand = inc @ uniq
                feasible = cand >= need
                if not feasible.any():
                    continue
                newcov = inc @ (~cur).astype(np.int8)
                score = np.where(feasible, newcov, -1)
                br = int(np.argmax(score))
                if br == bi or score[br] <= 0:
                    continue
                in_use[bi] = 0
                in_use[br] = 1
                recompute()
                if int(uncovered.sum()) == 0:
                    improved = True
                else:
                    in_use[br] = 0
                    in_use[bi] = 1
                    recompute()

    if time.time() - t0 < _DEADLINE - 15:
        local_search()

    # ---- Build final block list ----------------------------------------
    final_idx = np.where(in_use == 1)[0]
    blocks = [list(k_blocks[i]) for i in final_idx]

    # ---- Validate and guarantee coverage -------------------------------
    if len(blocks) > 0:
        fin = np.zeros(n_k, dtype=np.int8)
        for i in final_idx:
            fin[i] = 1
        covered = (inc.T @ fin) > 0
        if not covered.all():
            uncovered = (~covered).astype(np.int8)
            while int(uncovered.sum()) > 0 and time.time() - t0 < _DEADLINE - 1:
                cov = inc @ uncovered
                best = int(np.argmax(cov))
                if cov[best] == 0:
                    break
                blocks.append(list(k_blocks[best]))
                uncovered[inc[best] == 1] = 0
    else:
        blocks = []

    # Deduplicate
    seen = set()
    uniq = []
    for b in blocks:
        tb = tuple(sorted(b))
        if tb not in seen:
            seen.add(tb)
            uniq.append(list(tb))
    return uniq
