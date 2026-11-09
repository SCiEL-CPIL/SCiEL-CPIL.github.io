"""Fresh design for the C(24,7,3) covering task: fast bitset greedy + block-removal local search."""
import itertools
import random
import time


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before the 150s kill
    rng = random.Random(12345)

    # --- precompute t-subset <-> index mapping -------------------------------
    all_tsubs = list(itertools.combinations(range(v), t))
    n = len(all_tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    nbits = (n + 63) // 64

    # --- precompute, for each (point, (t-1)-tuple), the t-subset bitmask ------
    # a t-subset is covered iff it contains the (t-1)-tuple plus the new point
    comb_t1 = list(itertools.combinations(range(v), t - 1))
    pt_tuple_masks = {}
    for x in range(v):
        for tup in comb_t1:
            if x in tup:
                continue
            ts = tuple(sorted(tup + (x,)))
            m = 0
            for i in range(nbits):
                base = i * 64
                w = 0
                for b in range(64):
                    idx = base + b
                    if idx < n:
                        if tsub_idx[all_tsubs[idx]] == tsub_idx[ts]:
                            w |= 1 << b
                m |= w << (i * 64)
            pt_tuple_masks[(x, tup)] = m

    def greedy(seed, max_blocks=400):
        """Greedy set cover; returns (blocks, uncovered_mask)."""
        full = (1 << n) - 1
        uncovered = full
        blocks = []
        while uncovered and len(blocks) < max_blocks:
            # pick an uncovered t-subset as the seed for this block
            u = uncovered
            pos = 0
            while (u & ((1 << 64) - 1)) == 0:
                u >>= 64
                pos += 64
            block = list(all_tsubs[pos])

            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for tup in itertools.combinations(block, t - 1):
                        gain += (pt_tuple_masks[(x, tup)] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # remove every t-subset contained in this block
            for ts in itertools.combinations(block, t):
                uncovered &= ~(1 << tsub_idx[ts])
        return blocks, uncovered

    def removal_pass(blocks, max_rounds=200):
        """Drop blocks that are fully redundant; rebuild coverage each round."""
        cur = list(blocks)
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            cov = 0
            for b in cur:
                m = 0
                for ts in itertools.combinations(b, t):
                    m |= 1 << tsub_idx[ts]
                cov |= m
            full = (1 << n) - 1
            if cov != full:
                break
            progressed = False
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                m = 0
                for ts in itertools.combinations(cur[i], t):
                    m |= 1 << tsub_idx[ts]
                if (cov ^ m) == cov:  # every t-subset of this block is covered elsewhere
                    cov ^= m
                    cur.pop(i)
                    progressed = True
                    break
            if not progressed:
                break
        return cur

    best = None
    for seed in range(0, 4000):
        if time.time() > deadline - 1.0:
            break
        blocks, unc = greedy(seed)
        if unc == 0:
            blocks = removal_pass(blocks)
            if best is None or len(blocks) < len(best):
                best = blocks
    if best is not None:
        return best
    # fallback: deterministic greedy (should be complete)
    blocks, _ = greedy(0)
    return blocks
