"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

import numpy as np


def _precompute_blocks(v, k):
    """Generate all C(v,k) blocks and their covered t-subsets."""
    all_blocks = []
    block_triples = []
    for combo in itertools.combinations(range(v), k):
        all_blocks.append(combo)
        block_triples.append(frozenset(itertools.combinations(combo, 3)))
    return all_blocks, block_triples


def _greedy_set_cover(uncovered_set, block_triples, all_blocks, v, k, t, deadline):
    """Greedy set cover: repeatedly pick the block covering the most uncovered triples."""
    # Map each triple to the set of blocks that cover it
    triple_to_blocks = defaultdict(set)
    for i, btriples in enumerate(block_triples):
        for tr in btriples:
            triple_to_blocks[tr].add(i)

    # Count how many uncovered triples each block covers
    block_cover_count = np.zeros(len(all_blocks), dtype=np.int32)
    for tr in uncovered_set:
        for bi in triple_to_blocks.get(tr, ()):
            block_cover_count[bi] += 1

    blocks_used = []
    covered = set()
    n_blocks = len(all_blocks)

    while uncovered_set and time.time() < deadline:
        # Find block with max coverage
        best_idx = -1
        best_count = 0
        # Sample a subset of blocks for speed if there are many
        candidates = list(range(n_blocks))
        if len(candidates) > 5000:
            candidates = random.sample(candidates, 5000)
        for bi in candidates:
            if block_cover_count[bi] > best_count:
                best_count = block_cover_count[bi]
                best_idx = bi

        if best_idx == -1 or best_count == 0:
            break

        block = all_blocks[best_idx]
        blocks_used.append(block)

        # Update: mark newly covered triples
        new_covered = []
        for tr in block_triples[best_idx]:
            if tr in uncovered_set:
                uncovered_set.discard(tr)
                new_covered.append(tr)

        # Update block_cover_count for blocks that share these triples
        for tr in new_covered:
            for bi in triple_to_blocks.get(tr, ()):
                block_cover_count[bi] -= 1
            block_cover_count[best_idx] = 0  # This block is done

    return blocks_used


def _remove_redundant_blocks(blocks, v, t):
    """Remove blocks that don't cover any unique triple."""
    if not blocks:
        return blocks

    # For each triple, count how many blocks cover it
    triple_count = defaultdict(int)
    for block in blocks:
        for tr in itertools.combinations(block, t):
            triple_count[tr] += 1

    # A block is redundant if all its triples are covered by other blocks
    # Check: for each block, does it have at least one triple that only it covers?
    essential = []
    for block in blocks:
        is_essential = False
        for tr in itertools.combinations(block, t):
            if triple_count[tr] == 1:
                is_essential = True
                break
        if is_essential:
            essential.append(block)
        else:
            # Decrement counts for this block's triples
            for tr in itertools.combinations(block, t):
                triple_count[tr] -= 1

    return essential


def _local_search(blocks, all_blocks, block_triples, v, k, t, deadline):
    """Try to swap out a block for one that covers more new triples."""
    if not blocks:
        return blocks

    # Build current coverage
    covered = set()
    for block in blocks:
        for tr in itertools.combinations(block, t):
            covered.add(tr)

    # Map triple -> set of block indices in current solution
    triple_to_block_idx = defaultdict(set)
    for i, block in enumerate(blocks):
        for tr in itertools.combinations(block, t):
            triple_to_block_idx[tr].add(i)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each block and see if coverage is maintained
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if block i is essential
            essential = False
            for tr in itertools.combinations(blocks[i], t):
                if len(triple_to_block_idx[tr]) == 1:
                    essential = True
                    break
            if not essential:
                # Remove block i
                for tr in itertools.combinations(blocks[i], t):
                    triple_to_block_idx[tr].discard(i)
                blocks.pop(i)
                improved = True
                break

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Precompute all blocks and their triples
    all_blocks, block_triples = _precompute_blocks(v, k)
    n_blocks = len(all_blocks)

    # All triples
    all_triples = set(itertools.combinations(range(v), t))

    # Run multiple greedy restarts with randomization
    best_blocks = None
    best_len = float('inf')

    # First: deterministic greedy
    uncovered = set(all_triples)
    blocks = _greedy_set_cover(uncovered, block_triples, all_blocks, v, k, t, deadline)
    blocks = _remove_redundant_blocks(blocks, v, t)
    blocks = _local_search(blocks, all_blocks, block_triples, v, k, t, deadline)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = list(blocks)

    # Randomized restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break

        # Randomized greedy: perturb the choice
        uncovered = set(all_triples)
        blocks = []
        n_blocks_total = len(all_blocks)

        # Build triple-to-blocks map once
        triple_to_blocks = defaultdict(set)
        for i, btriples in enumerate(block_triples):
            for tr in btriples:
                triple_to_blocks[tr].add(i)

        block_cover_count = np.zeros(n_blocks_total, dtype=np.int32)
        for tr in uncovered:
            for bi in triple_to_blocks.get(tr, ()):
                block_cover_count[bi] += 1

        while uncovered and time.time() < deadline - 5.0:
            # Find top candidates
            nonzero = np.nonzero(block_cover_count > 0)[0]
            if len(nonzero) == 0:
                break

            # Pick from top candidates with some randomness
            counts = block_cover_count[nonzero]
            max_count = counts.max()
            top_candidates = nonzero[counts >= max_count - max(1, max_count // 20)]

            if len(top_candidates) > 50:
                # Sample to limit computation
                top_candidates = random.sample(top_candidates.tolist(), 50)

            best_idx = random.choice(top_candidates)

            block = all_blocks[best_idx]
            blocks.append(block)

            new_covered = []
            for tr in block_triples[best_idx]:
                if tr in uncovered:
                    uncovered.discard(tr)
                    new_covered.append(tr)

            for tr in new_covered:
                for bi in triple_to_blocks.get(tr, ()):
                    block_cover_count[bi] -= 1
            block_cover_count[best_idx] = 0

        blocks = _remove_redundant_blocks(blocks, v, t)
        blocks = _local_search(blocks, all_blocks, block_triples, v, k, t, deadline - 5.0)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(all_triples)
        best_blocks = []
        while uncovered and time.time() < deadline:
            best_block = None
            best_count = 0
            for block in all_blocks:
                count = sum(1 for tr in itertools.combinations(block, t) if tr in uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None or best_count == 0:
                break
            best_blocks.append(best_block)
            for tr in itertools.combinations(best_block, t):
                uncovered.discard(tr)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
