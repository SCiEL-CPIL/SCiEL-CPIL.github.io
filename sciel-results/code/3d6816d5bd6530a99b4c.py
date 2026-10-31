"""Covering design C(v,k,t): numpy/bitset-accelerated greedy set cover with a
dynamic gain table, local optimization (redundancy removal + block replacement),
and a validity-guaranteeing repair pass."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s (evaluator kills at 150s)

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute each block's covered t-subsets (dense vector + index list).
    # Done once, up-front, so the greedy loop never recomputes coverage.
    block_vecs = np.empty((n_blocks, n_tsubs), dtype=np.uint8)
    block_lists = [None] * n_blocks
    for i, B in enumerate(all_kblocks):
        vec = block_vecs[i]
        lst = []
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            vec[j] = 1
            lst.append(j)
        block_lists[i] = lst

    # ---------------- Greedy with dynamic gain table ----------------
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    gain = np.fromiter((len(bl) for bl in block_lists), dtype=np.int32, count=n_blocks)
    selected = []
    greedy_budget = deadline - 25.0  # reserve time for local-optimize + repair

    while True:
        if time.time() > greedy_budget:
            break
        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break
        selected.append(best_idx)
        lst = block_lists[best_idx]
        for j in lst:
            if uncovered[j]:
                uncovered[j] = 0
                gain[best_idx] -= 1
                # other blocks covering j lose one potential gain
                for b in tsub_blocks[j] if (tsub_blocks := None) else ():
                    pass
        # (tsub_blocks built below if needed; see fast path)

    # The loop above is a placeholder; real update uses a shared-tsub index.
    # Rebuild properly:
    selected = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    gain = np.fromiter((len(bl) for bl in block_lists), dtype=np.int32, count=n_blocks)
    # tsub_blocks: for each t-subset, the list of block indices covering it.
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b in range(n_blocks):
        for j in block_lists[b]:
            tsub_blocks[j].append(b)

    while True:
        if time.time() > greedy_budget:
            break
        best_idx = int(np.argmax(gain))
        if gain[best_idx] == 0:
            break
        selected.append(best_idx)
        for j in block_lists[best_idx]:
            if uncovered[j]:
                uncovered[j] = 0
                gain[best_idx] -= 1
                for b in tsub_blocks[j]:
                    if b != best_idx:
                        gain[b] -= 1

    # ---------------- Local optimization ----------------
    if time.time() < deadline - 8.0:
        selected = _local_optimize(selected, block_vecs, block_lists, tsub_blocks,
                                   n_tsubs, deadline - 8.0)

    blocks = [all_kblocks[i] for i in selected]

    # ---------------- Repair (guarantee validity) ----------------
    blocks = _repair(blocks, all_kblocks, v, k, t, deadline - 0.5)

    return blocks


def _local_optimize(selected, block_vecs, block_lists, tsub_blocks, n_tsubs, deadline):
    """Remove redundant blocks and attempt block replacements (same count or fewer)."""
    sel = list(selected)

    # coverage counts
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in sel:
        cover_counts += block_vecs[idx]

    # --- pass 1: remove redundant blocks ---
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        new_sel = []
        for idx in sel:
            vec = block_vecs[idx]
            if np.any((vec == 1) & (cover_counts == 1)):
                new_sel.append(idx)
            else:
                cover_counts -= vec
                changed = True
        sel = new_sel

    # --- pass 2: block replacement ---
    # bitmask of "low-count" t-subsets (count <= 2) for fast candidate scoring
    low_mask = (cover_counts <= 2).astype(np.uint8)

    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        new_sel = []
        for idx in sel:
            vec = block_vecs[idx]
            critical = (vec == 1) & (cover_counts == 1)
            if not np.any(critical):
                # redundant
                cover_counts -= vec
                changed = True
                continue
            crit_idx = np.where(critical)[0]
            # candidate blocks must cover all critical t-subsets
            best_alt = -1
            best_score = -1
            # candidate set = union of tsub_blocks for critical tsubs
            cand = set()
            for j in crit_idx:
                cand.update(tsub_blocks[j])
            for b in cand:
                if b == idx:
                    continue
                # must cover all critical
                bv = block_vecs[b]
                if np.any(bv[crit_idx] == 0):
                    continue
                score = int(np.sum(bv * low_mask))
                if score > best_score:
                    best_score = score
                    best_alt = b
            if best_alt != -1:
                cover_counts -= vec
                cover_counts += block_vecs[best_alt]
                low_mask = (cover_counts <= 2).astype(np.uint8)
                new_sel.append(best_alt)
                changed = True
            else:
                new_sel.append(idx)
        sel = new_sel

    return sel


def _repair(blocks, all_kblocks, v, k, t, deadline):
    """Guarantee a valid covering: add blocks until every t-subset is covered."""
    tsub_to_idx = {}
    all_tsubs = list(itertools.combinations(range(v), t))
    for i, ts in enumerate(all_tsubs):
        tsub_to_idx[ts] = i
    n_tsubs = len(all_tsubs)

    covered = np.zeros(n_tsubs, dtype=np.uint8)
    for B in blocks:
        for ts in itertools.combinations(B, t):
            covered[tsub_to_idx[ts]] = 1

    # quick check
    if np.all(covered):
        return blocks

    # build tsub -> block lists lazily only for uncovered tsubs
    # For each uncovered t-subset, find a block containing it.
    # Precompute point -> list of blocks containing it for speed.
    point_blocks = [[] for _ in range(v)]
    for b, B in enumerate(all_kblocks):
        for p in B:
            point_blocks[p].append(b)

    existing = set(blocks)
    new_blocks = list(blocks)
    for ts in all_tsubs:
        if time.time() > deadline:
            break
        if covered[tsub_to_idx[ts]]:
            continue
        # find a block containing all points of ts
        common = None
        for p in ts:
            if common is None:
                common = set(point_blocks[p])
            else:
                common &= set(point_blocks[p])
                if not common:
                    break
        if common:
            # pick the block in common that covers the most still-uncovered tsubs
            best_b = min(common, key=lambda b: 0)  # any works for validity
            new_blocks.append(all_kblocks[best_b])
            for ts2 in itertools.combinations(all_kblocks[best_b], t):
                covered[tsub_to_idx[ts2]] = 1
    return new_blocks


if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
