"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before the 150s hard kill

    # ------------------------------------------------------------------
    # Candidate pool: all nonzero weight-2, weight-3 and weight-4 vectors
    # with +/-1 entries.  Max norm squared = 4 (weight 4).
    # Pairwise feasibility:  dist_sq >= 4  <=>  n_i + n_j - 2*dot >= 4
    # ------------------------------------------------------------------
    candidates = []
    norms_sq = []

    for w in (2, 3, 4):
        for positions in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidates.append(v)
                norms_sq.append(w)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Conflict matrix:  dist_sq < 4  means the pair violates the contract.
    dots = np.dot(candidates, candidates.T)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = (dist_sq < 4)
    np.fill_diagonal(conflicts, False)

    # Fast per-vertex conflict bitmasks (Python ints) and neighbor sets.
    conf_list = [conflicts[i] for i in range(n_candidates)]
    neighbor_sets = [set(np.where(conf_list[i])[0].tolist()) for i in range(n_candidates)]
    conflict_masks = [0] * n_candidates
    for i in range(n_candidates):
        m = 0
        for j in neighbor_sets[i]:
            m |= (1 << j)
        conflict_masks[i] = m

    def greedy_min_degree(seed):
        """Randomized greedy: repeatedly pick a low-degree feasible vertex."""
        rng = random.Random(seed)
        available = set(range(n_candidates))
        selected = []
        while available and time.time() < deadline - 2.0:
            if len(available) > 400:
                sample = rng.sample(list(available), 400)
            else:
                sample = list(available)
            best_v = None
            min_deg = float("inf")
            for v in sample:
                deg = len(neighbor_sets[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v is None:
                break
            selected.append(best_v)
            available.discard(best_v)
            for n in neighbor_sets[best_v]:
                if n in available:
                    available.discard(n)
        return selected

    def local_search(S):
        """1-for-k swap + fill local search to escape greedy local minima."""
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 4.0:
            improved = False

            # 1) Fill: add any vertex compatible with the current set.
            for v in range(n_candidates):
                if time.time() > deadline - 4.0:
                    break
                if v not in S_set and not (neighbor_sets[v] & S_set):
                    S_set.add(v)
                    improved = True

            # 2) 1-for-k swaps: drop one vertex, greedily repack, keep if larger.
            if not improved:
                for v in list(S_set):
                    if time.time() > deadline - 4.0:
                        break
                    test_set = S_set - {v}
                    avail_swap = set()
                    for u in range(n_candidates):
                        if u in test_set or u == v:
                            continue
                        if not (neighbor_sets[u] & test_set):
                            avail_swap.add(u)
                    if not avail_swap:
                        continue
                    added = []
                    while avail_swap and len(added) < 4:
                        if time.time() > deadline - 4.0:
                            break
                        if len(avail_swap) > 60:
                            sample = random.sample(list(avail_swap), 60)
                        else:
                            sample = list(avail_swap)
                        best_u = None
                        min_deg = float("inf")
                        for u in sample:
                            deg = len(neighbor_sets[u] & avail_swap)
                            if deg < min_deg:
                                min_deg = deg
                                best_u = u
                        if best_u is None:
                            break
                        added.append(best_u)
                        avail_swap.discard(best_u)
                        for n in neighbor_sets[best_u]:
                            if n in avail_swap:
                                avail_swap.discard(n)
                    if len(added) > 1:  # net size gain
                        S_set = test_set | set(added)
                        improved = True

        return list(S_set)

    # ------------------------------------------------------------------
    # Phase 1: randomized greedy restarts (bulk of the time).
    # ------------------------------------------------------------------
    random.seed(42)
    np.random.seed(42)
    best = []
    greedy_deadline = start_time + (deadline - start_time) * 0.55
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        sel = greedy_min_degree(42 + trial)
        if len(sel) > len(best):
            best = sel

    # ------------------------------------------------------------------
    # Phase 2: local search on the best greedy result, then a few more
    # restarts refined with local search while time remains.
    # ------------------------------------------------------------------
    if best and time.time() < deadline - 6.0:
        refined = local_search(best)
        if len(refined) > len(best):
            best = refined

    extra_deadline = deadline - 6.0
    extra = 0
    while time.time() < extra_deadline:
        extra += 1
        seed = 100000 + extra
        sel = greedy_min_degree(seed)
        sel = local_search(sel)
        if len(sel) > len(best):
            best = sel

    if not best:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    return candidates[best].astype(np.int64)

# EVOLVE-BLOCK-END
