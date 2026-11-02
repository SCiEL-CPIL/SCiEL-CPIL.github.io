"""Covering design C(24,7,3) – structural local-search solver.

Strategy:
  1. Fast greedy construction using a NumPy-backed coverage counter for every
     triple (2024 triples).  Each 7-block covers C(7,3)=35 triples.
  2. Local search: repeatedly try to *remove* a block and re-cover the
     triples it uniquely covered by *replacing* it with a different block
     (or by redistributing).  In practice we do a simpler but effective
     move: swap one block for another that covers the same "critical" triples
     plus more, reducing total block count or enabling a later removal.
  3. A more powerful move: try to delete a block whose every covered triple
     is already covered by another block (redundant block removal).  Then
     try to find a block that, when added, allows deletion of two or more
     existing blocks (net gain).
  4. Deadline guard: stop at ~115 s and return the best valid covering found.
"""
import itertools
import time
import random
import numpy as np

_DEADLINE_FRACTION = 0.92  # stop at 92 % of the 120 s wall-clock budget


def _triple_index(a, b, c):
    """Map a triple (a,b,c) with 0<=a<b<c<v to a flat index in [0, C(v,3))."""
    # We precompute a lookup table for v=24.
    # index = number of triples with first element < a
    #       + number of triples (a, b', c') with b' < b
    #       + number of triples (a, b, c') with c' < c
    # For v=24:
    #   first(a) = sum_{i=0}^{a-1} C(24-1-i, 2)
    #   second(a,b) = sum_{j=a+1}^{b-1} (24-1-j)
    #   third(a,b,c) = c - b - 1
    # Precompute first and second as 2D arrays.
    return _TRIPLE_IDX[a, b, c]


# Precompute the triple-index table for v=24
def _build_triple_table(v=24):
    # _TRIPLE_IDX[a][b][c] for 0<=a<b<c<v
    table = np.zeros((v, v, v), dtype=np.int32)
    idx = 0
    for a in range(v - 2):
        for b in range(a + 1, v - 1):
            for c in range(b + 1, v):
                table[a, b, c] = idx
                idx += 1
    return table, idx


_TRIPLE_IDX, _NUM_TRIPLES = _build_triple_table(24)


def _block_triples(block):
    """Return the list of triple-indices covered by a 7-element block."""
    tris = []
    for i in range(len(block)):
        for j in range(i + 1, len(block)):
            for k in range(j + 1, len(block)):
                tris.append(_TRIPLE_IDX[block[i], block[j], block[k]])
    return tris


def _greedy_construct(v, k, t, rng):
    """
    Greedy set-cover: at each step pick the block that covers the most
    currently-uncovered triples.  Use a coverage counter array of length
    C(v,t).  Returns (blocks, coverage_count_array).
    """
    n_triples = _NUM_TRIPLES
    coverage = np.zeros(n_triples, dtype=np.int32)
    blocks = []

    # Precompute all C(v,k) blocks and their triple-lists for v=24, k=7.
    # C(24,7) = 346104 – feasible to precompute triple lists (35 ints each).
    all_blocks = list(itertools.combinations(range(v), k))
    # Precompute triple index lists for every block
    block_triples = []
    for blk in all_blocks:
        tris = []
        for i in range(k):
            for j in range(i + 1, k):
                for m in range(j + 1, k):
                    tris.append(_TRIPLE_IDX[blk[i], blk[j], blk[m]])
        block_triples.append(tris)

    # For speed, convert to numpy arrays
    block_triples_np = [np.array(tr, dtype=np.int32) for tr in block_triples]

    uncovered_count = n_triples
    while uncovered_count > 0:
        # Find the block that covers the most uncovered triples
        best_gain = -1
        best_idx = -1
        # Sub-sample for speed if too slow; but C(24,7)=346104, each has 35 entries.
        # 346104 * 35 ≈ 12.1M ops per iteration – too slow in Python.
        # Instead, use a smarter approach: iterate over uncovered triples and
        # accumulate a score for each candidate block.
        #
        # Alternative: maintain for each triple the set of blocks covering it.
        # That's too memory-heavy.
        #
        # Better: use the fact that we only need to find a good block, not the
        # optimal one.  Use a randomized greedy: pick a random uncovered triple,
        # then among blocks containing it, pick the one with highest gain.
        #
        # Let's precompute for each triple, the list of block indices that cover it.
        # C(24,3)=2024 triples, each covered by C(24-3,7-3)=C(21,4)=5985 blocks.
        # 2024 * 5985 ≈ 12.1M entries – feasible as a list of arrays.

        if not hasattr(_greedy_construct, '_triple_to_blocks'):
            triple_to_blocks = [[] for _ in range(n_triples)]
            for bi, tris in enumerate(block_triples_np):
                for tr in tris:
                    triple_to_blocks[tr].append(bi)
            _greedy_construct._triple_to_blocks = triple_to_blocks

        triple_to_blocks = _greedy_construct._triple_to_blocks

        # Pick a random uncovered triple
        # Find an uncovered triple: coverage[tr] == 0
        # Use a random sample
        while True:
            tr = rng.randint(0, n_triples - 1)
            if coverage[tr] == 0:
                break

        candidates = triple_to_blocks[tr]
        # Among candidates, pick the one with max gain
        best_gain = -1
        best_idx = -1
        for bi in candidates:
            gain = 0
            for tr2 in block_triples_np[bi]:
                if coverage[tr2] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = bi

        blk = all_blocks[best_idx]
        blocks.append(list(blk))
        for tr2 in block_triples_np[best_idx]:
            if coverage[tr2] == 0:
                uncovered_count -= 1
            coverage[tr2] += 1

    return blocks, coverage


def _is_valid(blocks, v, t):
    """Check that every t-subset is covered by at least one block."""
    coverage = np.zeros(_NUM_TRIPLES, dtype=np.int32)
    for blk in blocks:
        for i in range(len(blk)):
            for j in range(i + 1, len(blk)):
                for m in range(j + 1, len(blk)):
                    coverage[_TRIPLE_IDX[blk[i], blk[j], blk[m]]] += 1
    return np.all(coverage > 0)


def _local_search(blocks, v, k, t, rng, deadline):
    """
    Local search: try to reduce block count.
    Move 1: Remove a redundant block (all its triples covered by others).
    Move 2: Replace two blocks with one block that covers their union of
            uniquely-covered triples (if such a block exists).
    Move 3: Add a block, then remove two blocks (net -1).
    """
    n_triples = _NUM_TRIPLES
    blocks = [list(b) for b in blocks]

    # Precompute triple lists for current blocks
    def recompute_coverage(blks):
        cov = np.zeros(n_triples, dtype=np.int32)
        for blk in blks:
            for i in range(len(blk)):
                for j in range(i + 1, len(blk)):
                    for m in range(j + 1, len(blk)):
                        cov[_TRIPLE_IDX[blk[i], blk[j], blk[m]]] += 1
        return cov

    coverage = recompute_coverage(blocks)
    best_blocks = list(blocks)
    best_len = len(blocks)
    no_improve = 0

    while time.time() < deadline:
        improved = False

        # Move 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if block i is redundant
                blk = blocks[i]
                redundant = True
                for a in range(len(blk)):
                    for b in range(a + 1, len(blk)):
                        for c in range(b + 1, len(blk)):
                            tr = _TRIPLE_IDX[blk[a], blk[b], blk[c]]
                            if coverage[tr] <= 1:
                                redundant = False
                                break
                        if not redundant:
                            break
                    if not redundant:
                        break
                if redundant:
                    # Remove block i
                    for a in range(len(blk)):
                        for b in range(a + 1, len(blk)):
                            for c in range(b + 1, len(blk)):
                                coverage[_TRIPLE_IDX[blk[a], blk[b], blk[c]]] -= 1
                    blocks.pop(i)
                    changed = True
                    improved = True
                    break  # restart scan

        if len(blocks) < best_len:
            best_blocks = list(blocks)
            best_len = len(blocks)
            no_improve = 0
        else:
            no_improve += 1

        # Move 3: Try to add a block that allows removing 2+ blocks
        if no_improve >= 3:
            # Try random addition + removal
            all_7_blocks = list(itertools.combinations(range(v), k))
            # Sample a few candidate blocks
            for _ in range(50):
                if time.time() > deadline:
                    break
                cand = list(rng.sample(all_7_blocks, 1))[0]
                # Compute how many new triples this block covers
                # and how many blocks become redundant
                new_cov = coverage.copy()
                for a in range(k):
                    for b in range(a + 1, k):
                        for c in range(b + 1, k):
                            new_cov[_TRIPLE_IDX[cand[a], cand[b], cand[c]]] += 1

                # Count blocks that become redundant
                removable = 0
                for i in range(len(blocks)):
                    blk = blocks[i]
                    is_removable = True
                    for a in range(len(blk)):
                        for b in range(a + 1, len(blk)):
                            for c in range(b + 1, len(blk)):
                                tr = _TRIPLE_IDX[blk[a], blk[b], blk[c]]
                                if new_cov[tr] <= 1:
                                    is_removable = False
                                    break
                            if not is_removable:
                                break
                        if not is_removable:
                            break
                    if is_removable:
                        removable += 1

                if removable >= 2:
                    # Apply: add cand, remove the removable blocks
                    for a in range(k):
                        for b in range(a + 1, k):
                            for c in range(b + 1, k):
                                coverage[_TRIPLE_IDX[cand[a], cand[b], cand[c]]] += 1
                    blocks.append(list(cand))
                    # Remove removable blocks (scan from end)
                    for i in range(len(blocks) - 1, -1, -1):
                        if i == len(blocks) - 1 and blocks[i] == list(cand):
                            continue
                        blk = blocks[i]
                        is_removable = True
                        for a in range(len(blk)):
                            for b in range(a + 1, len(blk)):
                                for c in range(b + 1, len(blk)):
                                    tr = _TRIPLE_IDX[blk[a], blk[b], blk[c]]
                                    if coverage[tr] <= 1:
                                        is_removable = False
                                        break
                                if not is_removable:
                                    break
                            if not is_removable:
                                break
                        if is_removable:
                            for a in range(len(blk)):
                                for b in range(a + 1, len(blk)):
                                    for c in range(b + 1, len(blk)):
                                        coverage[_TRIPLE_IDX[blk[a], blk[b], blk[c]]] -= 1
                            blocks.pop(i)
                            removable -= 1
                        if removable <= 0:
                            break
                    improved = True
                    no_improve = 0
                    break

    if len(blocks) < len(best_blocks):
        best_blocks = list(blocks)
    return best_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 120.0 * _DEADLINE_FRACTION
    rng = random.Random(42)

    # Phase 1: Greedy construction
    blocks, coverage = _greedy_construct(v, k, t, rng)

    # Phase 2: Local search to reduce block count
    blocks = _local_search(blocks, v, k, t, rng, deadline)

    # Final validation
    if not _is_valid(blocks, v, t):
        # Fallback: re-run greedy (should not happen)
        blocks, _ = _greedy_construct(v, k, t, rng)

    return blocks
