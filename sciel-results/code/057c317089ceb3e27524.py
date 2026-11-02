"""
Covering design C(24,7,3) solver.

Strategy:
1. Fast greedy construction: pick whole blocks that maximize newly-covered t-subsets.
   Use a coverage-count array over all C(24,3)=2024 triples for speed.
2. Local search: iteratively try to remove a block if its t-subsets are all covered
   by other blocks; also try swapping one point in a block for another to reduce
   total block count or improve coverage.
3. Time-budgeted: return best valid covering before deadline.
"""
import itertools
import time
import random

# Precompute all t-subsets (triples) for v=24, t=3
_ALL_TRIPLES = None
_TRIPLE_INDEX = None  # map from frozenset -> index
_TRIPLE_SETS = None   # list of frozensets


def _init_triples(v, t):
    global _ALL_TRIPLES, _TRIPLE_INDEX, _TRIPLE_SETS
    if _ALL_TRIPLES is not None and v == 24 and t == 3:
        return
    _ALL_TRIPLES = list(itertools.combinations(range(v), t))
    _TRIPLE_SETS = [frozenset(tr) for tr in _ALL_TRIPLES]
    _TRIPLE_INDEX = {tr: i for i, tr in enumerate(_ALL_TRIPLES)}


def _triple_idx(tr):
    return _TRIPLE_INDEX[tr]


def _block_triples(block):
    """Return list of triple indices covered by a block (list of k points)."""
    return [_TRIPLE_INDEX[tuple(sorted(c))] for c in itertools.combinations(block, 3)]


def construct(v, k, t):
    _init_triples(v, t)
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    n_triples = len(_ALL_TRIPLES)
    # coverage[i] = number of blocks covering triple i
    coverage = [0] * n_triples
    blocks = []
    block_triples_list = []  # parallel: for each block, list of triple indices it covers

    def is_covered():
        return all(c > 0 for c in coverage)

    def add_block(block):
        blocks.append(block)
        tlist = _block_triples(block)
        block_triples_list.append(tlist)
        for ti in tlist:
            coverage[ti] += 1

    def remove_block(idx):
        tlist = block_triples_list[idx]
        for ti in tlist:
            coverage[ti] -= 1
        blocks.pop(idx)
        block_triples_list.pop(idx)

    # --- Phase 1: Greedy construction ---
    # For speed, we maintain a set of uncovered triple indices.
    uncovered = set(range(n_triples))

    while uncovered:
        if time.time() > deadline:
            break

        # Find the block that covers the most uncovered triples.
        # Heuristic: pick an uncovered triple, then try to build a block around it.
        # For efficiency, we sample a few uncovered triples and pick the best block.
        sample_size = min(20, len(uncovered))
        sample = random.sample(list(uncovered), sample_size)

        best_gain = -1
        best_block = None

        for tr_idx in sample:
            tr = _ALL_TRIPLES[tr_idx]
            # Try all k-subsets containing this triple that maximize uncovered coverage
            # This is expensive; instead, greedily build a block.
            block = list(tr)
            # Greedily add points to maximize new uncovered triples
            for _ in range(k - t):
                best_pt = -1
                best_g = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many new uncovered triples this point adds
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_tr = tuple(sorted(rest + (x,)))
                        if _TRIPLE_INDEX.get(new_tr, -1) in uncovered:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_pt = x
                if best_pt >= 0:
                    block.append(best_pt)
                else:
                    break
            block.sort()
            # Compute actual gain
            gain = 0
            for ti in _block_triples(block):
                if ti in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            # Fallback: just take the first uncovered triple and extend arbitrarily
            tr = _ALL_TRIPLES[next(iter(uncovered))]
            best_block = list(tr)
            for x in range(v):
                if len(best_block) >= k:
                    break
                if x not in best_block:
                    best_block.append(x)
            best_block.sort()

        add_block(best_block)
        # Update uncovered
        for ti in block_triples_list[-1]:
            if coverage[ti] == 1:
                uncovered.discard(ti)

    # If we didn't finish greedy in time, we still have a partial covering.
    # Ensure we have a valid covering by adding blocks for remaining uncovered.
    while uncovered:
        if time.time() > deadline:
            break
        tr = _ALL_TRIPLES[next(iter(uncovered))]
        block = list(tr)
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block:
                block.append(x)
        block.sort()
        add_block(block)
        for ti in block_triples_list[-1]:
            if coverage[ti] == 1:
                uncovered.discard(ti)

    # --- Phase 2: Local search to minimize block count ---
    # Try removing blocks that are redundant (all their triples covered by others).
    # Then try 1-point swaps to see if we can reduce block count.
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Try removing each block
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            tlist = block_triples_list[i]
            redundant = True
            for ti in tlist:
                if coverage[ti] <= 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                improved = True

        # Try 1-point swaps: replace one point in a block with another,
        # then check if any block becomes redundant.
        if len(blocks) > 1:
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                block = blocks[i]
                for pos in range(k):
                    old_pt = block[pos]
                    for new_pt in range(v):
                        if new_pt == old_pt or new_pt in block:
                            continue
                        new_block = list(block)
                        new_block[pos] = new_pt
                        new_block.sort()
                        new_tlist = _block_triples(new_block)

                        # Check if after swap, all triples in new_block are still covered
                        # by other blocks or by this new block
                        # And check if any other block becomes redundant
                        # For simplicity, just check if the swap doesn't leave any triple uncovered
                        ok = True
                        for ti in block_triples_list[i]:
                            if coverage[ti] <= 1 and ti not in new_tlist:
                                ok = False
                                break
                        if not ok:
                            continue

                        # Perform swap temporarily
                        old_tlist = block_triples_list[i]
                        for ti in old_tlist:
                            coverage[ti] -= 1
                        for ti in new_tlist:
                            coverage[ti] += 1
                        blocks[i] = new_block
                        block_triples_list[i] = new_tlist

                        # Check if any block is now redundant
                        for j in range(len(blocks)):
                            if j == i:
                                continue
                            tlist_j = block_triples_list[j]
                            redundant = True
                            for ti in tlist_j:
                                if coverage[ti] <= 1:
                                    redundant = False
                                    break
                            if redundant:
                                remove_block(j)
                                improved = True
                                break
                        else:
                            # No block removed, revert swap
                            for ti in new_tlist:
                                coverage[ti] -= 1
                            for ti in old_tlist:
                                coverage[ti] += 1
                            blocks[i] = block
                            block_triples_list[i] = old_tlist
                        break

    # Final validation: ensure every triple is covered
    # If not, add blocks for uncovered triples
    uncovered_final = set()
    for i in range(n_triples):
        if coverage[i] == 0:
            uncovered_final.add(i)

    while uncovered_final and time.time() < deadline:
        tr = _ALL_TRIPLES[next(iter(uncovered_final))]
        block = list(tr)
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block:
                block.append(x)
        block.sort()
        add_block(block)
        for ti in block_triples_list[-1]:
            if coverage[ti] == 1:
                uncovered_final.discard(ti)

    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
