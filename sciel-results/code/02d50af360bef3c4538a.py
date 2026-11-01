"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def _precompute_tsubsets(v, k, t):
    """Precompute all t-subsets of each potential block and useful indices."""
    # We'll use a different approach: precompute for each point, the list of (t-1)-subsets
    # that it can extend to form a t-subset. But since v=19 is small, let's just
    # precompute all C(19,5) = 11628 t-subsets and for each, the C(19,8) blocks that contain it.
    # Actually that's too many. Let's think differently.
    #
    # Key insight: For the greedy, when we have a partial block B (size < k), adding point x
    # covers t-subsets of the form S ∪ {x} where S is a (t-1)-subset of B.
    # So for each (t-1)-subset S of B, we need to know how many uncovered t-subsets are of form S ∪ {x}.
    #
    # Precompute: for each (t-1)-subset S (as a frozenset or tuple), a dict mapping x -> count of uncovered t-subsets S∪{x}
    # But this changes as we add blocks. So we need dynamic counts.
    #
    # Alternative: maintain a coverage array indexed by t-subset.
    # Number of t-subsets: C(19,5) = 11628.
    # For each t-subset, store how many blocks cover it.
    # When adding a block, for each t-subset in the block, increment its count.
    # When checking gain of adding x to partial block B: for each (t-1)-subset S of B,
    #   check if S∪{x} is an uncovered t-subset (count == 0).
    #
    # To speed up: precompute all t-subsets and map them to indices.
    # For each point x, precompute all t-subsets containing x.
    
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each point, list of t-subset indices that contain it
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for x in ts:
            point_to_tsubs[x].append(i)
    
    # For each (t-1)-subset, list of points x such that S∪{x} is a t-subset, and the index
    # This is for fast gain computation
    # (t-1)-subsets: C(19,4) = 3876
    tminus1_subs = list(itertools.combinations(range(v), t-1))
    tminus1_to_tsub_info = {}  # frozenset(S) -> list of (x, tsub_idx)
    for S in tminus1_subs:
        Sf = frozenset(S)
        if Sf not in tminus1_to_tsub_info:
            tminus1_to_tsub_info[Sf] = []
        for x in range(v):
            if x not in S:
                tsub = tuple(sorted(S + (x,)))
                if tsub in tsub_to_idx:
                    tminus1_to_tsub_info[Sf].append((x, tsub_to_idx[tsub]))
    
    return all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info


def _greedy_cover_fast(v, k, t, rng=None, deadline=None, precomp=None):
    """Fast greedy set-cover using precomputed data structures."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = float('inf')
    
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    
    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * n_tsubs
    blocks = []
    uncovered_count = n_tsubs
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick a random uncovered t-subset
        # Find uncovered indices
        uncovered_indices = [i for i in range(n_tsubs) if coverage[i] == 0]
        if not uncovered_indices:
            break
        seed_idx = rng.choice(uncovered_indices)
        seed = list(all_tsubs[seed_idx])
        block = list(seed)
        block_set = set(block)
        
        # Fill block greedily
        while len(block) < k:
            # For each candidate x not in block, compute gain
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Gain = number of uncovered t-subsets of form S ∪ {x} where S is (t-1)-subset of block
                gain = 0
                for S in itertools.combinations(block, t-1):
                    Sf = frozenset(S)
                    for (xx, idx) in tminus1_to_tsub_info[Sf]:
                        if xx == x and coverage[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(sorted(ts))]
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1
    
    return blocks, coverage


def _build_coverage(blocks, v, t, precomp):
    """Map t-subset index -> set of block indices that contain it."""
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    coverage = [set() for _ in range(n_tsubs)]
    for bi, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(sorted(ts))]
            coverage[idx].add(bi)
    return coverage


def _remove_block_fast(blocks, bi, coverage, t, precomp):
    """Remove block bi, update coverage. Returns newly uncovered t-subset indices."""
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    block = blocks[bi]
    newly_uncovered = []
    for ts in itertools.combinations(block, t):
        idx = tsub_to_idx[tuple(sorted(ts))]
        if len(coverage[idx]) == 1 and bi in coverage[idx]:
            newly_uncovered.append(idx)
        coverage[idx].discard(bi)
    blocks.pop(bi)
    # Re-index
    for idx in range(n_tsubs):
        if coverage[idx]:
            coverage[idx] = set(idx2 if idx2 < bi else idx2 - 1 for idx2 in coverage[idx])
    return newly_uncovered


def _add_block_fast(blocks, block, coverage, t, precomp):
    """Add a block, update coverage. Returns index of new block."""
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    bi = len(blocks)
    blocks.append(block)
    for ts in itertools.combinations(block, t):
        idx = tsub_to_idx[tuple(sorted(ts))]
        coverage[idx].add(bi)
    return bi


def _greedy_fill_fast(block, uncovered_set, v, k, t, precomp):
    """Fill block up to size k by greedily adding points that cover most uncovered t-subsets."""
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    block_set = set(block)
    while len(block) < k:
        best_x = None
        best_gain = -1
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for S in itertools.combinations(block, t-1):
                Sf = frozenset(S)
                for (xx, idx) in tminus1_to_tsub_info[Sf]:
                    if xx == x and idx in uncovered_set:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None or best_gain == 0:
            break
        block.append(best_x)
        block_set.add(best_x)
    return block


def _local_search_fast(blocks, v, k, t, deadline, precomp):
    """Local search to reduce block count."""
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp
    blocks = [list(b) for b in blocks]
    coverage = _build_coverage(blocks, v, t, precomp)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase A: Remove redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            newly_uncovered = []
            for ts in itertools.combinations(blocks[bi], t):
                idx = tsub_to_idx[tuple(sorted(ts))]
                if len(coverage[idx]) == 1 and bi in coverage[idx]:
                    newly_uncovered.append(idx)
            if not newly_uncovered:
                _remove_block_fast(blocks, bi, coverage, t, precomp)
                improved = True
                break

        if improved or time.time() >= deadline:
            continue

        # Phase B: Try swap
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            saved_blocks = [list(b) for b in blocks]
            saved_coverage = [set(s) for s in coverage]

            newly_uncovered = _remove_block_fast(blocks, bi, coverage, t, precomp)
            if not newly_uncovered:
                blocks = saved_blocks
                coverage = saved_coverage
                continue

            uncovered = set(newly_uncovered)
            new_blocks_added = 0
            max_new_blocks = 2

            while uncovered and new_blocks_added < max_new_blocks:
                if time.time() >= deadline:
                    break
                seed_idx = next(iter(uncovered))
                block = list(all_tsubs[seed_idx])
                block = _greedy_fill_fast(block, uncovered, v, k, t, precomp)
                block.sort()
                _add_block_fast(blocks, block, coverage, t, precomp)
                new_blocks_added += 1
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx[tuple(sorted(ts))]
                    uncovered.discard(idx)

            if len(blocks) < len(saved_blocks) and not uncovered:
                improved = True
                break
            else:
                blocks = saved_blocks
                coverage = saved_coverage

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    rng = random.Random(42)

    # Precompute
    precomp = _precompute_tsubsets(v, k, t)
    all_tsubs, tsub_to_idx, n_tsubs, point_to_tsubs, tminus1_to_tsub_info = precomp

    best_blocks = None
    best_count = float('inf')

    # Phase 1: Greedy cover with multiple restarts
    blocks, _ = _greedy_cover_fast(v, k, t, rng, deadline, precomp)
    best_blocks = blocks
    best_count = len(blocks)

    restarts = 0
    while time.time() < deadline - 45 and restarts < 10:
        blocks, _ = _greedy_cover_fast(v, k, t, rng, deadline, precomp)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search
    if time.time() < deadline - 10:
        best_blocks = _local_search_fast(best_blocks, v, k, t, deadline - 3, precomp)

    return best_blocks
# EVOLVE-BLOCK-END
