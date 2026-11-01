"""Covering design C(24,7,3): minimize 7-subsets covering all 2024 triples.

Strategy: vectorized greedy set-cover construction + block-deletion/reconstruction
local search. All heavy loops are NumPy-vectorized so each greedy step is O(v)
with a single large multiply, permitting many randomized restarts within budget.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # ---- Precompute triple index table (t=3) ----
    # triple_idx[a, b, c] = index of {a,b,c} in [0, N) for a<b<c, else -1
    triple_idx = np.full((v, v, v), -1, dtype=np.int64)
    all_triples = []
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                idx = len(all_triples)
                all_triples.append((a, b, c))
                triple_idx[a, b, c] = idx
    N = len(all_triples)  # C(24,3) = 2024

    # ---- Precompute, for each 7-point block, the flat array of its 35 triple indices ----
    # block_triples[i] = np.array of 35 ints (indices into [0,N))
    block_triples = {}
    for blk in itertools.combinations(range(v), k):
        arr = np.array(
            [triple_idx[a, b, c] for a in range(k) for b in range(a + 1, k)
             for c in range(b + 1, k)],
            dtype=np.int64,
        )
        block_triples[blk] = arr
    # Also a list of all block tuples for random restarts / candidate search
    all_blocks = list(itertools.combinations(range(v), k))

    # ---- Greedy construction (vectorized scoring) ----
    def greedy_construct(rng, seed_block=None):
        """Build a covering by greedy set-cover. Returns list of blocks (tuples)."""
        covered = np.zeros(N, dtype=np.int64)
        blocks = []
        if seed_block is not None:
            covered[block_triples[seed_block]] += 1
            blocks.append(seed_block)
        while True:
            if time.time() > deadline:
                break
            # gain[b] = number of currently-uncovered triples in block b
            # = 35 - sum over its triples of covered>0
            gains = np.empty(len(all_blocks), dtype=np.int64)
            # vectorized: for each block, count how many of its 35 triples are uncovered
            for bi, blk in enumerate(all_blocks):
                idxs = block_triples[blk]
                gains[bi] = np.count_nonzero(covered[idxs] == 0)
            # pick best gain; random tie-break among top
            maxg = gains.max()
            if maxg <= 0:
                break
            cands = np.where(gains == maxg)[0]
            bi = int(rng.choice(cands))
            blk = all_blocks[bi]
            covered[block_triples[blk]] += 1
            blocks.append(blk)
        return blocks

    def is_covering(covered):
        return bool((covered > 0).all())

    # ---- Deletion / reconstruction local search ----
    def local_search(blocks, rng):
        """Try to reduce block count by deleting and rebuilding. Returns improved list."""
        blocks = list(blocks)
        covered = np.zeros(N, dtype=np.int64)
        for blk in blocks:
            covered[block_triples[blk]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(blocks)
            # random order of block indices to try deletion
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline:
                    return blocks
                if i >= len(blocks):
                    continue
                blk = blocks[i]
                # temporarily remove
                covered[block_triples[blk]] -= 1
                # find triples now uncovered
                newly = np.where(covered == 0)[0]
                if len(newly) == 0:
                    # fully redundant: keep it removed
                    blocks.pop(i)
                    improved = True
                    continue
                # greedily build minimal set of blocks to cover `newly`
                # (only need to cover the newly-uncovered ones)
                need = set(newly.tolist())
                added = []
                while need and time.time() < deadline:
                    # for each candidate block, count how many of `need` it covers
                    best_bi = -1
                    best_g = 0
                    for bi, cb in enumerate(all_blocks):
                        if cb in blocks or cb in added:
                            continue
                        idxs = block_triples[cb]
                        g = 0
                        for ix in idxs:
                            if ix in need:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_bi = bi
                    if best_bi < 0 or best_g <= 0:
                        break
                    cb = all_blocks[best_bi]
                    added.append(cb)
                    for ix in block_triples[cb]:
                        need.discard(int(ix))
                if not need:
                    # successful: replace 1 block with len(added)
                    if len(added) < 1:
                        pass
                    else:
                        blocks.pop(i)
                        blocks.extend(added)
                        for nb in added:
                            covered[block_triples[nb]] += 1
                        if len(added) < 1:
                            improved = True
                    # net change = len(added) - 1
                    if len(added) < 1:
                        improved = True
                    # if len(added) == 1 we swapped (no count change) -> still may help later
                else:
                    # couldn't cover, restore
                    covered[block_triples[blk]] += 1
        return blocks

    # ---- Main: randomized restarts + local search, keep best ----
    rng = random.Random(1234)
    best_blocks = None
    best_count = float('inf')

    # First: a couple of quick greedy restarts to get a valid baseline fast
    for r in range(3):
        if time.time() > deadline:
            break
        blks = greedy_construct(rng)
        if len(blks) < best_count:
            best_blocks = blks
            best_count = len(blks)

    # Now iterate: local-search the best, then try fresh restarts in parallel budget
    for r in range(400):
        if time.time() > deadline:
            break
        # local search on current best
        if best_blocks is not None:
            ls = local_search(best_blocks, rng)
            if len(ls) < best_count:
                best_blocks = ls
                best_count = len(ls)
        # fresh greedy restart
        blks = greedy_construct(rng)
        if len(blks) < best_count:
            best_blocks = blks
            best_count = len(blks)
            # immediately polish the new best
            ls = local_search(best_blocks, rng)
            if len(ls) < best_count:
                best_blocks = ls
                best_count = len(ls)

    if best_blocks is None:
        # fallback: deterministic greedy
        best_blocks = greedy_construct(rng)

    # ---- Final polish on the best result ----
    if time.time() < deadline - 5.0:
        ls = local_search(best_blocks, rng)
        if len(ls) < best_count:
            best_blocks = ls
            best_count = len(ls)

    # Return as lists of ints
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
