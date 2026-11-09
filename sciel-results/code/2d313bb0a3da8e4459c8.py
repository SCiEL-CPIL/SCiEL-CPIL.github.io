"""Covering design C(v,k,t): frequency-greedy construction + iterative reduction, multi-start."""
import itertools
import time
import random
import sys
import numpy as np


def _tsub_index_map(v, t):
    """Map each t-subset (as a tuple) to a dense integer index."""
    combos = list(itertools.combinations(range(v), t))
    m = {}
    for i, c in enumerate(combos):
        m[c] = i
    return combos, m, len(combos)


def _block_tsub_indices(block, t, idx_map):
    """Return list of indices of t-subsets contained in block."""
    out = []
    for c in itertools.combinations(block, t):
        out.append(idx_map[c])
    return out


def _greedy_construct(v, k, t, combos, idx_map, n_t, rng, deadline):
    """Greedy: repeatedly add the k-block maximizing coverage of still-uncovered t-subsets."""
    # coverage count per t-subset (0 = uncovered)
    cov = np.zeros(n_t, dtype=np.int32)
    blocks = []
    while True:
        if time.time() > deadline:
            break
        if cov.min() >= 1:
            break
        # candidate: pick a block covering the most uncovered t-subsets
        # To keep it fast, choose a seed t-subset that is uncovered, then grow.
        uncovered = np.where(cov == 0)[0]
        if len(uncovered) == 0:
            break
        # pick a random uncovered t-subset as seed (randomization for multi-start)
        seed_idx = int(rng.choice(uncovered))
        seed = combos[seed_idx]
        block = list(seed)
        # grow block to size k by adding point with max gain in uncovered coverage
        block_set = set(block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # gain = number of (t-1)-subsets of current block that with x form an uncovered t-subset
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    c = tuple(sorted(rest + (x,)))
                    gi = idx_map[c]
                    if cov[gi] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        # apply: increment coverage for all t-subsets in block
        for gi in _block_tsub_indices(block, t, idx_map):
            cov[gi] += 1
        blocks.append(block)
    return blocks, cov


def _try_reduce(blocks, v, k, t, idx_map, n_t, deadline):
    """Iteratively remove blocks that are redundant. Returns reduced blocks."""
    blocks = list(blocks)
    if not blocks:
        return blocks
    
    # Precompute t-subset indices for each block
    block_tsubs = []
    for b in blocks:
        block_tsubs.append(_block_tsub_indices(b, t, idx_map))
    
    # coverage count per t-subset
    cov = np.zeros(n_t, dtype=np.int32)
    for tsubs in block_tsubs:
        for gi in tsubs:
            cov[gi] += 1
    
    # Track which t-subsets each block covers (for quick lookup)
    # For each block, store the list of t-subset indices it covers
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find all blocks that are redundant (no t-subset they cover has cov==1)
        removable = []
        for i, tsubs in enumerate(block_tsubs):
            if time.time() > deadline:
                break
            redundant = True
            for gi in tsubs:
                if cov[gi] == 1:
                    redundant = False
                    break
            if redundant:
                removable.append(i)
        
        # Remove all identified redundant blocks in one pass
        if removable:
            # Sort by index descending so we can remove safely
            removable.sort(reverse=True)
            for i in removable:
                tsubs = block_tsubs[i]
                for gi in tsubs:
                    cov[gi] -= 1
                del blocks[i]
                del block_tsubs[i]
            changed = True
    
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s kill

    combos, idx_map, n_t = _tsub_index_map(v, t)

    # Precompute per-block t-subset index lists cache is not trivial; recompute is fine.

    best = None
    best_len = None
    attempts = 0
    max_attempts = 40  # cap; but deadline governs

    # First do a deterministic greedy to get a valid baseline quickly.
    rng0 = random.Random(0)
    blocks0, _ = _greedy_construct(v, k, t, combos, idx_map, n_t, rng0, deadline)
    reduced0 = _try_reduce(blocks0, v, k, t, idx_map, n_t, deadline)
    best = reduced0
    best_len = len(reduced0)

    # Multi-start: randomize seed choice in greedy.
    seed = 1
    while time.time() < deadline - 10.0:
        rng = random.Random(seed)
        blocks, _ = _greedy_construct(v, k, t, combos, idx_map, n_t, rng, deadline)
        reduced = _try_reduce(blocks, v, k, t, idx_map, n_t, deadline)
        if len(reduced) < best_len:
            best = reduced
            best_len = len(reduced)
        seed += 1
        attempts += 1
        if attempts >= max_attempts:
            break

    # Final aggressive reduction pass on the best solution found
    if best and time.time() < deadline - 2.0:
        final_reduced = _try_reduce(best, v, k, t, idx_map, n_t, deadline)
        if len(final_reduced) < best_len:
            best = final_reduced
            best_len = len(final_reduced)

    # Final safety: ensure validity (should hold). Return best.
    return best
