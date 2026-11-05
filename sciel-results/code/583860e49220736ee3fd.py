"""Conflict-Aware Greedy covering for C(v,k,t) with time-budgeted polish."""
import time
import random
import itertools

import numpy as np


def _deadline_guard(deadline):
    return time.perf_counter() < deadline


def _gen_candidates(v, k, t, n_fixed, rng):
    """Generate a diverse pool of candidate k-subsets.

    Mix of:
      - random blocks
      - blocks anchored on the most 'lonely' t-subsets (points that appear
        together in few blocks so far)
      - blocks seeded from a small random core, then completed by a quick
        greedy that maximizes coverage of currently-uncovered t-subsets.
    """
    candidates = set()
    for _ in range(n_fixed):
        core = list(rng.sample(range(v), min(k, t)))
        # extend core greedily (cheap, only a few steps) to cover uncovered sets
        # We do a light greedy here using a rough scoring.
        block = list(core)
        # simple random completion
        remaining = [x for x in range(v) if x not in block]
        rng.shuffle(remaining)
        for x in remaining:
            if len(block) >= k:
                break
            block.append(x)
        if len(block) == k:
            candidates.add(tuple(sorted(block)))
    return candidates


def _block_tsets(block, t):
    return list(itertools.combinations(block, t))


def _build_index(v, k, t):
    """Map each t-subset -> list of block indices containing it (for polish)."""
    # We'll build lazily in polish.
    pass


def construct(v, k, t):
    t0 = time.perf_counter()
    deadline = t0 + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    # ---------- Phase 1: Conflict-Aware Greedy over candidate pool ----------
    # We maintain for each t-subset a count of how many selected blocks cover it.
    # A block's "new" coverage = number of t-subsets it covers with count==0.
    # A block's "total" coverage = number of t-subsets it covers (with count>=1).
    # Ratio = new / total penalizes redundancy.

    # Precompute the universe of t-subsets and assign an integer id.
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_id = {ts: i for i, ts in enumerate(all_tsets)}

    # count[i] = how many selected blocks cover t-subset i
    count = np.zeros(n_tsets, dtype=np.int32)
    uncovered_mask = np.ones(n_tsets, dtype=bool)  # count==0

    selected = []  # list of blocks (tuples)
    # block -> list of tset indices it covers
    selected_tsets = []  # parallel list

    # Candidate pool generation (diverse)
    n_pool = 3000
    pool = []
    # random blocks
    for _ in range(n_pool // 2):
        b = tuple(sorted(rng.sample(range(v), k)))
        pool.append(b)
    # blocks anchored on random t-subsets
    for _ in range(n_pool // 2):
        seed = rng.sample(range(v), t)
        rest = rng.sample([x for x in range(v) if x not in seed], k - t)
        b = tuple(sorted(seed + rest))
        pool.append(b)

    # Precompute tset index lists for each pool block
    pool_tsets = []
    for b in pool:
        pool_tsets.append([tset_id[ts] for ts in itertools.combinations(b, t)])

    # Greedy selection loop
    while True:
        if not _deadline_guard(deadline - 5.0):
            break
        # score each pool block: new / total
        best_score = -1.0
        best_idx = -1
        # vectorized: for each pool block compute new and total
        # To keep it fast, do a loop but with numpy per block (cheap).
        for idx, tlist in enumerate(pool_tsets):
            arr = np.asarray(tlist, dtype=np.int64)
            new = int(np.count_nonzero(uncovered_mask[arr]))
            total = len(tlist)
            if new <= 0:
                continue
            score = new / total
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            # no pool block adds new coverage; regenerate pool fresh
            # (uncovered sets remain; rebuild pool anchored on them)
            unc_idx = np.nonzero(uncovered_mask)[0]
            if len(unc_idx) == 0:
                break
            pool = []
            pool_tsets = []
            for _ in range(n_pool):
                anchor = all_tsets[int(unc_idx[rng.randrange(len(unc_idx))])]
                rest = rng.sample([x for x in range(v) if x not in anchor], k - t)
                b = tuple(sorted(list(anchor) + rest))
                pool.append(b)
                pool_tsets.append([tset_id[ts] for ts in itertools.combinations(b, t)])
            continue

        block = pool[best_idx]
        tlist = pool_tsets[best_idx]
        selected.append(block)
        selected_tsets.append(tlist)
        count[tlist] += 1
        # update uncovered_mask where count went 0->1
        for i in tlist:
            if count[i] == 1:
                uncovered_mask[i] = False
        if not uncovered_mask.any():
            break

    # ---------- Phase 2: Local-search polish (swap / drop) ----------
    if not _deadline_guard(deadline - 3.0):
        return [list(b) for b in selected]

    # Build index: tset -> list of selected block indices
    tset_to_blocks = [[] for _ in range(n_tsets)]
    for bi, tlist in enumerate(selected_tsets):
        for i in tlist:
            tset_to_blocks[i].append(bi)

    # count of coverage per tset
    cov = np.array([len(lst) for lst in tset_to_blocks], dtype=np.int32)

    def try_drop(bi):
        # can we remove block bi without uncovering anything?
        tlist = selected_tsets[bi]
        for i in tlist:
            if cov[i] <= 1:
                return False
        return True

    improved = True
    while improved and _deadline_guard(deadline - 3.0):
        improved = False
        # try dropping any redundant block
        for bi in range(len(selected) - 1, -1, -1):
            if try_drop(bi):
                # remove
                tlist = selected_tsets[bi]
                for i in tlist:
                    cov[i] -= 1
                    tset_to_blocks[i].remove(bi)
                selected.pop(bi)
                selected_tsets.pop(bi)
                improved = True
                break

    # Try 1-for-1 swaps: replace a block with a candidate that reduces count.
    # We attempt to find a block whose removal frees coverage, then a new block
    # that covers the freed t-sets. This is expensive; do a light pass.
    if _deadline_guard(deadline - 2.0) and len(selected) > 1:
        # Build a candidate pool for swaps anchored on currently single-covered sets.
        single = np.nonzero(cov == 1)[0]
        if len(single) > 0:
            # try removing a random single-cover block and see if a pool block covers all
            for attempt in range(200):
                if not _deadline_guard(deadline - 2.0):
                    break
                bi = rng.randrange(len(selected))
                tlist = selected_tsets[bi]
                # check all its tsets have cov>=2 after removal? need cov>=2 for each
                removable = all(cov[i] >= 2 for i in tlist)
                if not removable:
                    continue
                # find a pool block covering all of tlist
                ok = False
                for idx, pl in enumerate(pool_tsets):
                    if len(pl) < len(tlist):
                        continue
                    s = set(pl)
                    if all(i in s for i in tlist):
                        # accept swap: remove bi, add pool block
                        for i in tlist:
                            cov[i] -= 1
                            tset_to_blocks[i].remove(bi)
                        selected.pop(bi)
                        selected_tsets.pop(bi)
                        nb = pool[idx]
                        selected.append(nb)
                        selected_tsets.append(pl)
                        for i in pl:
                            cov[i] += 1
                            tset_to_blocks[i].append(len(selected) - 1)
                        ok = True
                        break
                if ok:
                    break

    # ---------- Final: ensure validity (every tset covered) ----------
    # Recompute coverage to be safe; if any gap, fill with greedy single blocks.
    final_cov = np.zeros(n_tsets, dtype=np.int32)
    for tlist in selected_tsets:
        final_cov[tlist] += 1
    gaps = np.nonzero(final_cov == 0)[0]
    for gi in gaps:
        ts = all_tsets[gi]
        rest = rng.sample([x for x in range(v) if x not in ts], k - t)
        b = tuple(sorted(list(ts) + rest))
        selected.append(b)
        selected_tsets.append([tset_id[x] for x in itertools.combinations(b, t)])

    # de-duplicate blocks (keep first occurrence)
    seen = set()
    unique = []
    for b in selected:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    selected = unique

    return [list(b) for b in selected]
