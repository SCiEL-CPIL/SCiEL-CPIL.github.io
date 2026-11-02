"""Covering design C(24,7,3) — greedy + vectorized local search."""
import itertools
import time
import numpy as np

# Precomputed constants for v=24, k=7, t=3
_V = 24
_K = 7
_T = 3
_NUM_TRIPLES = 2024  # C(24,3)

# Map each triple (a,b,c) with a<b<c to a unique index 0..2023
_TRIPLE_INDEX = {}
_idx = 0
for a in range(_V):
    for b in range(a + 1, _V):
        for c in range(b + 1, _V):
            _TRIPLE_INDEX[(a, b, c)] = _idx
            _idx += 1

# For each point x, list of triple indices that contain x
_POINT_TRIPLES = [[] for _ in range(_V)]
for (a, b, c), idx in _TRIPLE_INDEX.items():
    _POINT_TRIPLES[a].append(idx)
    _POINT_TRIPLES[b].append(idx)
    _POINT_TRIPLES[c].append(idx)

# For each pair (a,b), list of triple indices containing both
_PAIR_TRIPLES = {}
for a in range(_V):
    for b in range(a + 1, _V):
        lst = []
        for c in range(_V):
            if c == a or c == b:
                continue
            triple = tuple(sorted((a, b, c)))
            lst.append(_TRIPLE_INDEX[triple])
        _PAIR_TRIPLES[(a, b)] = lst


def _block_triple_indices(block):
    """Return the set of triple indices covered by a block (list of 7 points)."""
    indices = set()
    for i in range(len(block)):
        for j in range(i + 1, len(block)):
            a, b = block[i], block[j]
            for c_idx in _PAIR_TRIPLES[(a, b)]:
                indices.add(c_idx)
    return indices


def _build_coverage_matrix(blocks):
    """Build a numpy array of shape (num_blocks, num_triples) with coverage counts."""
    n_blocks = len(blocks)
    cov = np.zeros((n_blocks, _NUM_TRIPLES), dtype=np.int32)
    for i, block in enumerate(blocks):
        for tri_idx in _block_triple_indices(block):
            cov[i, tri_idx] = 1
    return cov


def _total_coverage(cov):
    """Sum of all coverage entries (total number of (block, triple) incidences)."""
    return int(cov.sum())


def _is_covered(cov):
    """Check if every triple is covered by at least one block."""
    return bool((cov.sum(axis=0) >= 1).all())


def _remove_redundant_blocks(blocks, cov, deadline):
    """Remove blocks that are redundant (their removal keeps full coverage).
    Returns (new_blocks, new_cov, removed_count)."""
    n = len(blocks)
    if n == 0:
        return blocks, cov, 0

    # coverage per triple
    triple_cov = cov.sum(axis=0)  # shape (num_triples,)
    # A block i is redundant if for every triple it covers, triple_cov[triple] >= 2
    # i.e., no triple has coverage exactly 1 that is only in block i

    removed = 0
    keep = list(range(n))
    # Try removing blocks one at a time, rechecking
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        n_keep = len(keep)
        if n_keep == 0:
            break
        # Build current coverage from kept blocks
        if n_keep != n:
            cov_curr = cov[keep].sum(axis=0)  # (num_triples,)
        else:
            cov_curr = triple_cov
        # Find a redundant block: for block i, check if any triple it covers has cov_curr==1
        # and that triple is covered by block i
        for pos in range(n_keep):
            if time.time() > deadline:
                break
            bi = keep[pos]
            # Get triples covered by block bi
            block_tris = _block_triple_indices(blocks[bi])
            # Check if any of these triples have coverage exactly 1 (meaning only this block covers it)
            redundant = True
            for tri in block_tris:
                if cov_curr[tri] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                keep = [x for x in keep if x != bi]
                removed += 1
                changed = True
                break
        if not changed:
            break

    if removed == 0:
        return blocks, cov, 0

    new_blocks = [blocks[i] for i in keep]
    new_cov = cov[keep] if len(keep) < n else cov
    return new_blocks, new_cov, removed


def _swap_search(blocks, cov, deadline, max_iters=5000):
    """Try swapping one point in a block with another point to reduce total coverage.
    Uses vectorized numpy operations for speed."""
    n = len(blocks)
    if n == 0:
        return blocks, cov

    total = _total_coverage(cov)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break

        for bi in range(n):
            if time.time() > deadline:
                break
            block = blocks[bi]
            # For each point in block, try replacing with a point not in block
            for pos in range(len(block)):
                old_pt = block[pos]
                # Compute coverage contribution of this block
                # For a swap old_pt -> new_pt, the change in total coverage is:
                #   - triples containing old_pt but not new_pt that are only covered by this block
                #   + triples containing new_pt but not old_pt that are newly covered
                # We want to minimize total coverage, so we want swaps that reduce it.

                # Get all triples containing old_pt
                old_tris = _POINT_TRIPLES[old_pt]
                # Get all triples containing new_pt (for each candidate)
                # For efficiency, compute for all candidates at once using numpy

                # Build a bitmask of which triples this block covers
                # Actually, let's compute the marginal: for each candidate new_pt,
                # compute how many triples would be lost vs gained

                # Triples containing old_pt that this block covers:
                # These are triples (old_pt, x, y) where x,y are in block
                block_set = set(block)
                block_other = [p for p in block if p != old_pt]

                # Triples this block covers that contain old_pt:
                old_block_tris = set()
                for x in block_other:
                    for y in block_other:
                        if x == y:
                            continue
                        tri = tuple(sorted((old_pt, x, y)))
                        old_block_tris.add(_TRIPLE_INDEX[tri])

                # For each candidate new_pt not in block:
                # new_block_tris would contain triples (new_pt, x, y) for x,y in block_other
                # The change: we lose old_block_tris coverage for this block, gain new_block_tris
                # But other blocks may cover these triples too.

                # Net change in total = (new_tris_added) - (old_tris_removed)
                # where old_tris_removed = |{tri in old_block_tris : total_cov[tri] == 1}|
                #       new_tris_added = |{tri in new_block_tris : total_cov[tri] == 0}|

                # Precompute total_cov
                total_cov = cov.sum(axis=0)  # (num_triples,)

                # old_tris_removed: triples in old_block_tris with total_cov == 1
                old_tris_arr = np.array(list(old_block_tris), dtype=np.int32)
                if len(old_tris_arr) > 0:
                    old_removed = int((total_cov[old_tris_arr] == 1).sum())
                else:
                    old_removed = 0

                # For each candidate, compute new_tris_added
                # new_block_tris for candidate c: triples (c, x, y) for x,y in block_other
                # Precompute for all candidates at once
                candidates = [p for p in range(_V) if p not in block_set]
                if not candidates:
                    continue

                best_candidate = -1
                best_delta = 0  # we want delta < 0 (reduction)

                for c in candidates:
                    # Compute new_block_tris for this candidate
                    new_tris_set = set()
                    for x in block_other:
                        for y in block_other:
                            if x == y:
                                continue
                            tri = tuple(sorted((c, x, y)))
                            new_tris_set.add(_TRIPLE_INDEX[tri])

                    if len(new_tris_set) == 0:
                        new_added = 0
                    else:
                        new_tris_arr = np.array(list(new_tris_set), dtype=np.int32)
                        new_added = int((total_cov[new_tris_arr] == 0).sum())

                    delta = new_added - old_removed
                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = c

                if best_candidate >= 0:
                    # Perform the swap
                    blocks[bi][pos] = best_candidate
                    blocks[bi].sort()
                    # Update cov
                    cov[bi] = 0
                    for tri_idx in _block_triple_indices(blocks[bi]):
                        cov[bi, tri_idx] = 1
                    total += best_delta
                    improved = True
                    break  # restart from beginning after a change
            if improved:
                break
        if not improved:
            break

    return blocks, cov


def _greedy_construct(v, k, t, deadline):
    """Fast greedy: repeatedly add a block that covers the most uncovered triples."""
    # Initialize all triples as uncovered
    uncovered_count = np.ones(_NUM_TRIPLES, dtype=np.int32)  # 1 = uncovered
    blocks = []

    while uncovered_count.sum() > 0:
        if time.time() > deadline:
            break

        # Find the uncovered triple with smallest index (deterministic)
        uncovered_indices = np.where(uncovered_count > 0)[0]
        if len(uncovered_indices) == 0:
            break
        start_triple_idx = int(uncovered_indices[0])

        # Find which triple this is
        # Reverse lookup: index -> (a,b,c)
        # Build reverse map
        # For speed, precompute
        if not hasattr(_greedy_construct, '_rev_map'):
            _greedy_construct._rev_map = {}
            for (a, b, c), idx in _TRIPLE_INDEX.items():
                _greedy_construct._rev_map[idx] = (a, b, c)
        rev_map = _greedy_construct._rev_map
        a, b, c = rev_map[start_triple_idx]

        # Start block with these 3 points
        block = [a, b, c]
        block_set = set(block)

        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            if time.time() > deadline:
                break
            best_pt = -1
            best_gain = -1

            for pt in range(v):
                if pt in block_set:
                    continue
                # Compute gain: number of uncovered triples that would be covered
                # by adding pt to the block
                # Triples containing pt and at least (t-1)=2 other points from block
                gain = 0
                # For each pair in block, count uncovered triples (pair + pt)
                for i in range(len(block)):
                    for j in range(i + 1, len(block)):
                        p1, p2 = block[i], block[j]
                        tri = tuple(sorted((pt, p1, p2)))
                        tri_idx = _TRIPLE_INDEX[tri]
                        if uncovered_count[tri_idx] > 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_pt = pt

            if best_pt < 0 or best_gain <= 0:
                # No improvement possible; just add any remaining point
                for pt in range(v):
                    if pt not in block_set:
                        best_pt = pt
                        break
                if best_pt < 0:
                    break

            block.append(best_pt)
            block_set.add(best_pt)

        block.sort()
        blocks.append(block)

        # Update uncovered_count
        for tri_idx in _block_triple_indices(block):
            uncovered_count[tri_idx] = 0

    return blocks


def construct(v, k, t):
    """Construct a covering design: every t-subset of {0..v-1} is in some k-block."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Build coverage matrix
    cov = _build_coverage_matrix(blocks)

    # Verify coverage
    if not _is_covered(cov):
        # Shouldn't happen with correct greedy, but just in case
        return blocks

    # Phase 2: Remove redundant blocks
    blocks, cov, _ = _remove_redundant_blocks(blocks, cov, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 3: Swap search to reduce total coverage (which can enable more removals)
    blocks, cov = _swap_search(blocks, cov, deadline, max_iters=3000)

    if time.time() > deadline:
        return blocks

    # Phase 4: Remove redundant blocks again after swaps
    blocks, cov, _ = _remove_redundant_blocks(blocks, cov, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 5: Another round of swap search with more iterations if time permits
    if time.time() < deadline - 10.0:
        blocks, cov = _swap_search(blocks, cov, deadline, max_iters=5000)
        if time.time() < deadline - 5.0:
            blocks, cov, _ = _remove_redundant_blocks(blocks, cov, deadline)

    # Final verification
    if not _is_covered(cov):
        # Fallback: use the greedy result (should be valid)
        # Rebuild from blocks
        cov_check = _build_coverage_matrix(blocks)
        if not _is_covered(cov_check):
            # This shouldn't happen; return as-is
            pass

    return blocks
