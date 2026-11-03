"""Covering design C(v,k,t) via whole-block greedy + beam search + local polish."""
import itertools
import time


def _build_index(v, k, t):
    pts = range(v)
    all_blocks = [tuple(c) for c in itertools.combinations(pts, k)]
    sub_list = [tuple(c) for c in itertools.combinations(pts, t)]
    sub_index = {c: j for j, c in enumerate(sub_list)}
    # block_mask[i]: bitmask of t-subsets covered by block i
    nsub = len(sub_list)
    block_mask = [0] * len(all_blocks)
    for i, b in enumerate(all_blocks):
        mask = 0
        for c in itertools.combinations(b, t):
            mask |= 1 << sub_index[c]
        block_mask[i] = mask
    # subset_to_blocks: sub -> list of block indices covering it
    subset_to_blocks = {c: [] for c in sub_list}
    for i, b in enumerate(all_blocks):
        for c in itertools.combinations(b, t):
            subset_to_blocks[c].append(i)
    return all_blocks, sub_list, block_mask, subset_to_blocks


def _redundancy_remove(blocks, subset_to_blocks, deadline):
    n = len(blocks)
    if n < 2:
        return blocks
    counts = [0] * n
    for c, idxs in subset_to_blocks.items():
        present = [i for i in idxs if i < n]
        if len(present) >= 2:
            for i in present:
                counts[i] += 1
    keep = [i for i in range(n) if counts[i] == 0]
    if len(keep) == n:
        return blocks
    kept = [blocks[i] for i in keep]
    for c, idxs in subset_to_blocks.items():
        if any(i in idxs for i in keep):
            continue
        best = -1
        for i in idxs:
            if i < n and best == -1:
                best = i
        kept.append(blocks[best])
    return kept


def _local_improve(blocks, subset_to_blocks, deadline):
    n = len(blocks)
    if n < 2:
        return blocks
    counts = [0] * n
    for c, idxs in subset_to_blocks.items():
        present = [i for i in idxs if i < n]
        if len(present) >= 2:
            for i in present:
                counts[i] += 1
    rem = [i for i in range(n) if counts[i] == 0]
    if not rem:
        return blocks
    for r in rem:
        if time.time() > deadline:
            break
        cand = [blocks[i] for i in range(n) if i != r]
        for c, idxs in subset_to_blocks.items():
            if r not in idxs:
                continue
            cands = [i for i in idxs if i < n and i != r]
            if not cands:
                break
            for b in cands:
                if b in cand:
                    continue
                trial = cand + [b]
                if len(trial) < n:
                    blocks = trial
                    n = len(blocks)
                    counts = [0] * n
                    for c2, idxs2 in subset_to_blocks.items():
                        present = [i for i in idxs2 if i < n]
                        if len(present) >= 2:
                            for i in present:
                                counts[i] += 1
                    rem = [i for i in range(n) if counts[i] == 0]
                    if not rem:
                        return blocks
                    break
            else:
                continue
            break
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 128.0  # safety margin under the 150s kill
    all_blocks, sub_list, block_mask, subset_to_blocks = _build_index(v, k, t)
    nsub = len(sub_list)
    all_mask = (1 << nsub) - 1

    # Whole-block greedy seed: pick the block covering the most uncovered t-subsets.
    covered = 0
    blocks = []
    while covered != all_mask:
        if time.time() > deadline:
            break
        best = -1
        best_gain = 0
        for i in range(len(all_blocks)):
            gain = bin(block_mask[i] & ~covered).count("1")
            if gain > best_gain:
                best_gain = gain
                best = i
        if best < 0 or best_gain == 0:
            break
        blocks.append(all_blocks[best])
        covered |= block_mask[best]

    best_blocks = blocks

    # Beam search over partial coverings.
    def popcount(x):
        return bin(x).count("1")

    beam = [(covered, list(range(len(blocks))))]
    for step in range(15):
        if time.time() > deadline:
            break
        next_states = []
        for cov, ids in beam:
            scored = []
            for i in range(len(all_blocks)):
                g = popcount(block_mask[i] & ~cov)
                if g > 0:
                    scored.append((g, i))
            scored.sort(reverse=True)
            for g, i in scored[:6]:
                next_states.append((cov | block_mask[i], ids + [i]))
        next_states.sort(key=lambda st: (len(st[1]), -popcount(st[0])))
        beam = next_states[:5]
        for cov, ids in beam:
            bl = [all_blocks[i] for i in ids]
            if len(bl) < len(best_blocks):
                best_blocks = bl
        if len(best_blocks) <= 320:
            break

    # Polish.
    best_blocks = _redundancy_remove(best_blocks, subset_to_blocks, deadline)
    best_blocks = _local_improve(best_blocks, subset_to_blocks, deadline)
    best_blocks = _redundancy_remove(best_blocks, subset_to_blocks, deadline)
    return best_blocks
