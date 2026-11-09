"""Covering design C(24,7,3) - optimized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    deadline = time.time() + 130.0
    rng = random.Random(42)

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute point masks (bitsets of triples each point is in)
    point_mask = []
    for p in range(v):
        mask = 0
        for ts in all_tsubs:
            if p in ts:
                mask |= (1 << tsub_to_idx[ts])
        point_mask.append(mask)

    full_mask = (1 << n_tsubs) - 1

    def popcount(x):
        return bin(x).count('1')

    def block_to_mask(block):
        m = 0
        for ts in itertools.combinations(block, t):
            m |= (1 << tsub_to_idx[ts])
        return m

    def greedy(seed):
        r = random.Random(seed)
        uncovered = full_mask
        blocks = []
        # Pre-shuffle triple order for tie-breaking
        order = list(range(n_tsubs))
        r.shuffle(order)

        while uncovered:
            # Find first uncovered triple in shuffled order
            target = None
            for idx in order:
                if (uncovered >> idx) & 1:
                    target = idx
                    break
            if target is None:
                break

            block = list(all_tsubs[target])
            cur_mask = 0
            for p in block:
                cur_mask |= point_mask[p]

            while len(block) < k:
                best_p = -1
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = popcount(uncovered & point_mask[x] & ~cur_mask)
                    if g > best_g:
                        best_g = g
                        best_p = x
                block.append(best_p)
                cur_mask |= point_mask[best_p]

            block.sort()
            blocks.append(block)
            uncovered &= ~block_to_mask(block)

        return blocks

    def valid(blocks):
        cov = 0
        for b in blocks:
            cov |= block_to_mask(b)
        return cov == full_mask

    def local_search(blocks, tlimit):
        if not valid(blocks):
            return blocks
        masks = [block_to_mask(b) for b in blocks]
        n = len(blocks)

        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > tlimit:
                    return blocks
                others = 0
                for j in range(len(blocks)):
                    if j != i:
                        others |= masks[j]
                if (masks[i] & ~others) == 0:
                    blocks.pop(i)
                    masks.pop(i)
                    changed = True
                    break

        # Try block replacements
        for _ in range(100):
            if time.time() > tlimit:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > tlimit:
                    break
                others = 0
                for j in range(len(blocks)):
                    if j != i:
                        others |= masks[j]
                unique = masks[i] & ~others
                if popcount(unique) > 8:
                    continue
                # Try replacement
                for _ in range(3):
                    if time.time() > tlimit:
                        break
                    # Pick random triple from block i
                    tris = list(itertools.combinations(blocks[i], t))
                    st = tris[rng.randint(0, len(tris) - 1)]
                    nb = list(st)
                    cm = 0
                    for p in nb:
                        cm |= point_mask[p]
                    while len(nb) < k:
                        bp = -1
                        bg = -1
                        for x in range(v):
                            if x in nb:
                                continue
                            g = popcount(unique & point_mask[x] & ~cm)
                            if g > bg:
                                bg = g
                                bp = x
                        if bp < 0:
                            break
                        nb.append(bp)
                        cm |= point_mask[bp]
                    nb.sort()
                    nm = block_to_mask(nb)
                    # Check validity
                    if (masks[i] & ~(others | nm)) == 0:
                        blocks[i] = nb
                        masks[i] = nm
                        improved = True
                        break
            if not improved:
                break
        return blocks

    best = None
    best_n = float('inf')

    for s in range(8):
        if time.time() > deadline - 50:
            break
        b = greedy(s)
        if valid(b) and len(b) < best_n:
            best = b
            best_n = len(b)

    if best is not None and time.time() < deadline - 25:
        lb = local_search(best, deadline - 10)
        if valid(lb) and len(lb) < best_n:
            best = lb
            best_n = len(lb)

    if best is None:
        best = greedy(0)

    return best
# EVOLVE-BLOCK-END
