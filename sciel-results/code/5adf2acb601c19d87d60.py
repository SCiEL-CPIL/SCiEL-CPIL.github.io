"""
Greedy set-cover for C(v,k,t) with precomputed block->triple maps,
multiple randomized restarts, redundancy pruning, and a hard deadline guard.
"""
import itertools
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
from collections import defaultdict, Counter

# Global cache for precomputed data
_cache = {}


def _get_precomputed(v, k, t):
    """Precompute all k-blocks and the triples they cover, plus triple->blocks index."""
    key = (v, k, t)
    if key in _cache:
        return _cache[key]

    all_blocks = list(itertools.combinations(range(v), k))
    # For each block, the set of t-subsets (as tuples) it covers
    block_triples = []
    for b in all_blocks:
        block_triples.append(set(itertools.combinations(b, t)))

    # Triple -> list of block indices
    triple_to_blocks = defaultdict(list)
    for idx, tris in enumerate(block_triples):
        for tr in tris:
            triple_to_blocks[tr].append(idx)

    # Total number of triples
    all_triples = list(itertools.combinations(range(v), t))

    _cache[key] = (all_blocks, block_triples, triple_to_blocks, all_triples)
    return _cache[key]


def _greedy_cover(v, k, t, rng=None, max_blocks=None, deadline=None):
    """
    Run one greedy set-cover pass.
    Returns a list of blocks (as tuples of ints).
    """
    if rng is None:
        rng = random.Random()

    all_blocks, block_triples, triple_to_blocks, all_triples = _get_precomputed(v, k, t)
    n_blocks = len(all_blocks)

    # uncovered: set of triple-tuples still not covered
    uncovered = set(all_triples)
    # covered_count[triple] = number of selected blocks that cover this triple
    covered_count = defaultdict(int)
    selected = []  # list of block indices
    selected_set = set()

    # We'll track for each block how many uncovered triples it covers (its "gain")
    # gain[block_idx] = number of uncovered triples in block_triples[block_idx]
    gain = [len(tr) for tr in block_triples]

    # To pick the best block quickly, we can use a heap or just scan.
    # For v=24,k=7: C(24,7)=346104 blocks. Scanning 346k per step is expensive.
    # But number of steps is at most ~76-80. 346k * 80 = ~27.7M operations per greedy run.
    # With multiple restarts this adds up. Let's use a smarter approach.

    # Alternative: maintain a priority structure.
    # For simplicity and to avoid complex data structures, let's use a two-level approach:
    # 1. For each uncovered triple, know which blocks can cover it.
    # 2. Pick the uncovered triple with the fewest candidate blocks (most constrained).
    # 3. Among blocks covering that triple, pick the one with highest gain.

    # Build: for each triple, list of block indices (already have triple_to_blocks)
    # But we need to filter to only unselected blocks.

    while uncovered:
        if deadline and time.time() > deadline:
            break
        if max_blocks and len(selected) >= max_blocks:
            break

        # Find the most constrained uncovered triple
        # (one with fewest candidate blocks that haven't been selected yet)
        # This is expensive to do naively. Let's use a heuristic:
        # Pick a random uncovered triple, or the one with min candidates.
        # For speed, let's pick a random sample of uncovered triples and pick the most constrained among them.

        sample_size = min(50, len(uncovered))
        sample = rng.sample(list(uncovered), sample_size) if len(uncovered) > sample_size else list(uncovered)

        best_triple = None
        min_candidates = float('inf')
        for tr in sample:
            candidates = [b for b in triple_to_blocks[tr] if b not in selected_set]
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_triple = tr
            if min_candidates == 1:
                break

        if best_triple is None:
            break

        # Among blocks covering best_triple, pick the one with max gain
        candidates = [b for b in triple_to_blocks[best_triple] if b not in selected_set]
        if not candidates:
            # Shouldn't happen if we're tracking correctly
            break

        best_block = max(candidates, key=lambda b: gain[b])

        # Select this block
        selected.append(best_block)
        selected_set.add(best_block)

        # Update uncovered and gains
        for tr in block_triples[best_block]:
            if tr in uncovered:
                uncovered.discard(tr)
            # Update gain for other blocks that share this triple
            # This is expensive. Let's just recompute gain lazily or skip.
            # Actually, updating all gains for each triple is O(number of blocks per triple) per triple,
            # which is too slow. Let's just not update gains incrementally and instead
            # recompute the best block's gain on the fly when needed.

        # Simpler: just recompute gain for the chosen block as 0 (it's selected now)
        gain[best_block] = 0

    # Convert to list of blocks (tuples)
    result = [all_blocks[i] for i in selected]
    return result


def _prune_redundant(blocks, v, k, t, deadline=None):
    """Remove blocks that are redundant (their triples are all covered by other blocks)."""
    if not blocks:
        return blocks

    all_triples = set(itertools.combinations(range(v), t))
    n = len(blocks)

    # For each block, compute the set of triples it covers
    block_triples = [set(itertools.combinations(b, t)) for b in blocks]

    # Check if each block is redundant
    kept = []
    for i in range(n):
        if deadline and time.time() > deadline:
            kept.append(blocks[i])
            continue

        # Check if block i's triples are all covered by other blocks
        redundant = True
        for tr in block_triples[i]:
            covered_by_other = False
            for j in range(n):
                if j != i and tr in block_triples[j]:
                    covered_by_other = True
                    break
            if not covered_by_other:
                redundant = False
                break

        if not redundant:
            kept.append(blocks[i])

    return kept


def _greedy_with_gain_update(v, k, t, rng=None, deadline=None):
    """
    Greedy set cover with proper gain updates.
    Uses a different strategy: always pick the block with max gain among all unselected blocks.
    This is simpler but may be slower due to scanning all blocks.
    """
    all_blocks, block_triples, triple_to_blocks, all_triples = _get_precomputed(v, k, t)
    n_blocks = len(all_blocks)

    uncovered = set(all_triples)
    selected = []
    selected_set = set()
    gain = [len(tr) for tr in block_triples]

    # For efficient max-gain selection, we could use a heap, but gains change.
    # Let's just scan. For v=24,k=7, n_blocks ~ 346k. Number of iterations ~ 76.
    # 346k * 76 = ~26M comparisons per run. Should be OK for a few restarts.

    while uncovered:
        if deadline and time.time() > deadline:
            break

        # Find block with max gain
        best_block = -1
        best_gain = 0
        for b in range(n_blocks):
            if b not in selected_set and gain[b] > best_gain:
                best_gain = gain[b]
                best_block = b

        if best_block == -1:
            break

        selected.append(best_block)
        selected_set.add(best_block)

        # Update gains: for each triple in this block, decrement gain of all other blocks covering it
        for tr in block_triples[best_block]:
            if tr in uncovered:
                uncovered.discard(tr)
            # Decrement gain for other blocks
            for other_b in triple_to_blocks[tr]:
                if other_b != best_block and other_b not in selected_set:
                    gain[other_b] -= 1

    return [all_blocks[i] for i in selected]


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    # Deadline: leave 5 seconds margin
    deadline = start_time + 115.0

    # First, try the gain-update greedy once as a baseline
    best_blocks = _greedy_with_gain_update(v, k, t, rng=random.Random(42), deadline=deadline)
    best_count = len(best_blocks)

    # Try multiple randomized restarts with the constrained-triple greedy
    rng = random.Random(123)
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        blocks = _greedy_cover(v, k, t, rng=rng, deadline=deadline)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        # Try pruning
        pruned = _prune_redundant(blocks, v, k, t, deadline=deadline)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = pruned

    # Also try pruning the best so far
    pruned_best = _prune_redundant(best_blocks, v, k, t, deadline=deadline)
    if len(pruned_best) < best_count:
        best_blocks = pruned_best
        best_count = len(pruned_best)

    # Final check: if we're still within time, do a few more gain-update greedy runs
    while time.time() < deadline - 2.0:
        blocks = _greedy_with_gain_update(v, k, t, rng=random.Random(), deadline=deadline)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            pruned = _prune_redundant(best_blocks, v, k, t, deadline=deadline)
            if len(pruned) < best_count:
                best_blocks = pruned
                best_count = len(pruned)

    # Return as list of lists
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
