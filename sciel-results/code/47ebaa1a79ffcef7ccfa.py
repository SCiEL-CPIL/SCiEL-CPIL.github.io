"""Fast bitmask greedy + local-search covering for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 145.0   # safety margin vs. 150s kill

    # ---- Precomputation -------------------------------------------------
    all_pts = range(v)
    tsubsets = list(itertools.combinations(all_pts, t))
    M = 1 << len(tsubsets)

    # mask_for[x] : bitmask of all t-subsets containing point x
    mask_for = [0] * v
    for i, ts in enumerate(tsubsets):
        bit = 1 << i
        for x in ts:
            mask_for[x] |= bit

    # ---- Greedy construction -------------------------------------------
    def build_block(start):
        block = [start]
        covered = mask_for[start]
        while len(block) < k:
            best_x, best_c = -1, -1
            for x in all_pts:
                if x in block:
                    continue
                c = (mask_for[x] & ~covered).bit_count()
                if c > best_c:
                    best_c, best_x = c, x
            block.append(best_x)
            covered |= mask_for[best_x]
        return block

    uncovered = M - 1
    blocks = []
    while uncovered and time.time() < deadline:
        lb = (uncovered & -uncovered).bit_length() - 1
        start = tsubsets[lb][0]
        b = build_block(start)
        blocks.append(b)
        bm = 0
        for x in b:
            bm |= mask_for[x]
        uncovered &= ~bm

    # ---- Local search --------------------------------------------------
    def block_mask(b):
        m = 0
        for x in b:
            m |= mask_for[x]
        return m

    def try_remove(blocks):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                m = block_mask(blocks[i])
                union = 0
                for j in range(len(blocks)):
                    if j != i:
                        union |= block_mask(blocks[j])
                if (m & ~union) == 0:
                    del blocks[i]
                    changed = True
                    break

    try_remove(blocks)

    # swap-based refinement
    while time.time() < deadline - 5.0:
        improved = False
        for i in range(len(blocks)):
            if time.time() >= deadline - 5.0:
                break
            b = blocks[i]
            for pos in range(k):
                if time.time() >= deadline - 5.0:
                    break
                cur = b[pos]
                rest = [x for j, x in enumerate(b) if j != pos]
                rest_mask = 0
                for x in rest:
                    rest_mask |= mask_for[x]
                best_x, best_c = cur, (mask_for[cur] & ~rest_mask).bit_count()
                for x in all_pts:
                    if x in rest:
                        continue
                    c = (mask_for[x] & ~rest_mask).bit_count()
                    if c > best_c:
                        best_c, best_x = c, x
                if best_x != cur:
                    b[pos] = best_x
                    b.sort()
                    improved = True
        try_remove(blocks)
        if not improved:
            break

    # final de-dup + sort
    seen = set()
    out = []
    for b in blocks:
        b = sorted(b)
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out
# EVOLVE-BLOCK-END
