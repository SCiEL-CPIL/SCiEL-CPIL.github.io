"""Fresh design for covering design C(v,k,t): global greedy set cover with
vectorized gain updates + redundancy minimalization, under a deadline guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _count_table(v, k, t):
    """Map each t-subset (as a tuple) to an integer index; return (table, n_t)."""
    table = {}
    n_t = 0
    for c in itertools.combinations(range(v), t):
        table[c] = n_t
        n_t += 1
    return table, n_t


def construct(v, k, t):
    start = time.time()
    deadline = start + 135.0  # safety margin vs 150s kill

    # --- precompute t-subset index table and per-point t-subset lists ---
    table, n_t = _count_table(v, k, t)
    ptsub = [[] for _ in range(v)]
    for c in itertools.combinations(range(v), t):
        idx = table[c]
        for x in c:
            ptsub[x].append(idx)

    # --- greedy set cover: each block covers its C(k,t) t-subsets ---
    covered = np.zeros(n_t, dtype=np.int8)
    blocks = []

    def add_block(bl):
        for c in itertools.combinations(bl, t):
            covered[table[c]] = 1

    while True:
        if time.time() > deadline:
            break
        unc = int(np.count_nonzero(covered == 0))
        if unc == 0:
            break
        # find the (k)-block with maximum uncovered gain
        best = None
        best_gain = -1
        for comb in itertools.combinations(range(v), k):
            g = 0
            for c in itertools.combinations(comb, t):
                if covered[table[c]] == 0:
                    g += 1
            if g > best_gain:
                best_gain = g
                best = comb
                if g == unc:  # perfect: covers everything remaining
                    break
        if best is None or best_gain <= 0:
            break
        bl = list(best)
        bl.sort()
        blocks.append(bl)
        add_block(bl)

    # --- minimalization: remove any block whose removal leaves coverage intact ---
    # precompute, for each block, the set of t-subset indices it covers
    block_sets = []
    for bl in blocks:
        s = set()
        for c in itertools.combinations(bl, t):
            s.add(table[c])
        block_sets.append(s)

    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        new_sets = []
        for i, s in enumerate(block_sets):
            # temporarily remove this block
            for idx in s:
                covered[idx] = 0
            ok = True
            for idx in s:
                if covered[idx] == 0:
                    ok = False
                    break
            if not ok:
                # keep it: restore
                for idx in s:
                    covered[idx] = 1
                new_blocks.append(blocks[i])
                new_sets.append(s)
            else:
                # safe to drop: leave covered==0 (other blocks still cover)
                changed = True
        blocks = new_blocks
        block_sets = new_sets

    return blocks
# EVOLVE-BLOCK-END
