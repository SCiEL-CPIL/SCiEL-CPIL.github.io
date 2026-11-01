"""Vectorized multi-restart greedy covering with local search."""
import itertools
import random
import time

import numpy as np


def _precompute(v, t):
    """Precompute all t-subsets and a mapping from (t-subset) to index."""
    tris = list(itertools.combinations(range(v), t))
    tri_idx = {tr: i for i, tr in enumerate(tris)}
    return tris, tri_idx


def _triple_indices_for_block(block, v, t, tri_idx, precomputed_block_tris=None):
    """Return list of triple indices covered by a given block."""
    return [tri_idx[tr] for tr in itertools.combinations(block, t)]


def _build_greedy(v, k, t, tris, tri_idx, rng, deadline, alpha=1.0):
    """
    Greedy covering with vectorized coverage counts.
    alpha: weight for random tie-breaking (higher = more random).
    Returns (blocks, coverage_counts, uncovered_count).
    """
    n_tris = len(tris)
    # coverage[i] = number of blocks covering triple i
    cov = np.zeros(n_tris, dtype=np.int32)
    uncovered = np.ones(n_tris, dtype=bool)
    n_uncovered = n_tris
    blocks = []

    while n_uncovered > 0 and time.time() < deadline:
        # Pick the first uncovered triple (deterministic anchor)
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        anchor_triple = tris[unc_idx[0]]
        block = list(anchor_triple)

        # Greedily fill the block
        while len(block) < k:
            in_block = set(block)
            candidates = [x for x in range(v) if x not in in_block]
            if not candidates:
                break

            # For each candidate, compute how many currently-uncovered triples it would cover
            # A triple (a,b,c) is covered by block+{x} iff it contains x and the other two are in block,
            # OR it's formed by x and two from block.
            # Actually: adding x to block covers all triples that contain x and (t-1) elements from block.
            # We need triples that are currently uncovered AND contain x AND contain (t-1) from block.

            best_gain = -1
            best_cands = []
            for x in candidates:
                # Count uncovered triples that contain x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    idx = tri_idx[tr]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [x]
                elif gain == best_gain:
                    best_cands.append(x)

            # Choose with random tie-breaking
            if best_cands:
                # Weighted random: prefer higher gain, but allow ties
                if alpha > 0 and len(best_cands) > 1:
                    # Pick randomly among best
                    x = rng.choice(best_cands)
                else:
                    x = best_cands[0]
            else:
                x = candidates[0]

            block.append(x)

        block.sort()
        blocks.append(block)

        # Update coverage
        for tr in itertools.combinations(block, t):
            idx = tri_idx[tr]
            if uncovered[idx]:
                cov[idx] += 1
                uncovered[idx] = False
                n_uncovered -= 1
            else:
                cov[idx] += 1

    return blocks, cov, n_uncovered


def _local_search(blocks, v, k, t, tri_idx, deadline, max_iters=200):
    """
    Try to reduce block count by:
    1. Removing redundant blocks (if a block's triples are all covered by others).
    2. Swapping points in blocks to increase coverage (then remove redundant).
    """
    blocks = [list(b) for b in blocks]
    n_tris = len(tri_idx)

    def compute_cov(block_list):
        cov = np.zeros(n_tris, dtype=np.int32)
        for b in block_list:
            for tr in itertools.combinations(b, t):
                cov[tri_idx[tr]] += 1
        return cov

    improved = True
    while improved and time.time() < deadline:
        improved = False
        cov = compute_cov(blocks)

        # Try removing each block if redundant
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if block i is redundant: all its triples have cov > 1
            redundant = True
            for tr in itertools.combinations(blocks[i], t):
                if cov[tri_idx[tr]] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                improved = True
                break

        if not improved:
            # Try swapping points in each block
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                block = blocks[i]
                # For each point in block, try replacing with a better one
                for pos in range(k):
                    if time.time() > deadline:
                        break
                    old_pt = block[pos]
                    # Compute coverage loss from removing old_pt
                    loss = 0
                    for tr in itertools.combinations(block, t):
                        if old_pt in tr and cov[tri_idx[tr]] == 1:
                            loss += 1
                    if loss > 0:
                        # This point is critical; skip or try to find replacement that maintains coverage
                        # For simplicity, skip critical points
                        continue

                    # Try each candidate replacement
                    best_gain = 0
                    best_x = None
                    for x in range(v):
                        if x in block:
                            continue
                        new_block = block[:pos] + [x] + block[pos+1:]
                        # Compute gain: new triples covered
                        gain = 0
                        for tr in itertools.combinations(new_block, t):
                            if x in tr:
                                if cov[tri_idx[tr]] == 0:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is not None and best_gain > 0:
                        block[pos] = best_x
                        block.sort()
                        improved = True
                        break
                if improved:
                    break

    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Safety margin: finish by 115s to allow overhead
    deadline = start_time + 110.0

    tris, tri_idx = _precompute(v, t)
    n_tris = len(tris)

    # Deterministic baseline first
    rng_det = random.Random(42)
    best_blocks, _, best_uncovered = _build_greedy(v, k, t, tris, tri_idx, rng_det, deadline)

    # If we have time, try multiple random restarts with different tie-breaking
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try several restarts with varying alpha
        alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        attempt = 0
        for alpha in alphas:
            for seed in seeds:
                if time.time() > deadline - 3.0:
                    break
                rng = random.Random(seed * 1000 + int(alpha * 10))
                blocks, _, unc = _build_greedy(v, k, t, tris, tri_idx, rng, deadline - 2.0, alpha=alpha)
                if len(blocks) < len(best_blocks):
                    best_blocks = blocks
                    best_uncovered = unc
                attempt += 1
            if time.time() > deadline - 3.0:
                break

    # Local search on the best
    if time.time() < deadline - 2.0:
        best_blocks = _local_search(best_blocks, v, k, t, tri_idx, deadline - 1.0)

    # Final validity check and cleanup
    # Verify every triple is covered
    final_cov = np.zeros(n_tris, dtype=np.int32)
    for b in best_blocks:
        for tr in itertools.combinations(b, t):
            final_cov[tri_idx[tr]] += 1
    if np.any(final_cov == 0):
        # Fallback: use the deterministic greedy result
        best_blocks, _, _ = _build_greedy(v, k, t, tris, tri_idx, random.Random(0), deadline)

    # Ensure blocks are distinct and sorted
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        tple = tuple(sorted(b))
        if tple not in seen:
            seen.add(tple)
            unique_blocks.append(list(tple))

    return unique_blocks
