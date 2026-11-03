# EVOLVE-BLOCK-START
import time
import itertools
import random

_TMAX = 150.0
_DEADLINE = _TMAX - 15.0  # safety margin before 150s kill


def construct(v, k, t):
    start = time.time()

    def elapsed():
        return time.time() - start

    deadline = start + _DEADLINE

    # ---- precompute t-subsets and per-point triple masks ----
    triples = list(itertools.combinations(range(v), t))
    total = len(triples)
    point_masks = [0] * v
    for i, tr in enumerate(triples):
        bit = 1 << i
        for p in tr:
            point_masks[p] |= bit
    all_mask = (1 << total) - 1
    uncovered = all_mask

    def block_mask(block):
        m = 0
        for p in block:
            m |= point_masks[p]
        return m

    # ---- greedy construction ----
    blocks = []
    while uncovered and elapsed() < deadline:
        low = uncovered & (-uncovered)
        i = low.bit_length() - 1
        start_triple = triples[i]
        block = list(start_triple)
        bm = 0
        for p in block:
            bm |= point_masks[p]
        while len(block) < k and elapsed() < deadline:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = (point_masks[x] & uncovered & ~bm).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            bm |= point_masks[best_x]
        block.sort()
        blocks.append(block)
        uncovered &= ~block_mask(block)

    if uncovered:
        # Could not cover in time; return what we have (may be invalid).
        return blocks

    # ---- local search to reduce block count ----
    def covers_all(blks):
        m = 0
        for b in blks:
            m |= block_mask(b)
        return m == all_mask

    cur = list(blocks)
    if not covers_all(cur):
        return cur

    rng = random.Random(20240612)

    def try_drop_replace(cur):
        """Try to remove one block by fixing another via a point swap."""
        n = len(cur)
        if n <= 1:
            return cur, False
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if elapsed() > deadline:
                break
            rest = cur[:i] + cur[i+1:]
            rest_mask = 0
            for b in rest:
                rest_mask |= block_mask(b)
            missing = all_mask & ~rest_mask
            if missing == 0:
                return rest, True
            # try to cover `missing` by swapping a point in some rest block
            for j in range(len(rest)):
                if elapsed() > deadline:
                    return cur, False
                bj = list(rest[j])
                bj_mask = block_mask(bj)
                for p in list(bj):
                    base_mask = bj_mask & ~point_masks[p]
                    for x in range(v):
                        if x in bj:
                            continue
                        new_mask = base_mask | point_masks[x]
                        new_rest_mask = (rest_mask & ~bj_mask) | new_mask
                        if (all_mask & ~new_rest_mask) == 0:
                            bj.remove(p)
                            bj.append(x)
                            bj.sort()
                            rest[j] = bj
                            return rest, True
        return cur, False

    improved = True
    while improved and elapsed() < deadline:
        improved = False
        cur, ok = try_drop_replace(cur)
        if ok:
            improved = True

    return cur


# EVOLVE-BLOCK-END
