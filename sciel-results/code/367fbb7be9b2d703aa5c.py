"""Covering design C(19,7,4) - optimized greedy with local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

# Precompute all 4-subsets for v=19
# C(19,4) = 3876
# Map each 4-subset to a bit index
# We'll use a set of frozensets for uncovered 4-subsets

def _precompute_all_tsubsets(v, t):
    """Return list of all t-subsets as tuples, and a dict mapping tuple->index."""
    all_subs = list(itertools.combinations(range(v), t))
    idx = {s: i for i, s in enumerate(all_subs)}
    return all_subs, idx


def _greedy_block(uncovered_set, block_seed, v, k, t, all_tsubs, tsub_idx):
    """Build a block starting from block_seed, adding points that cover most uncovered t-subsets."""
    block = list(block_seed)
    # Precompute: for each point x not in block, how many uncovered t-subsets would it cover
    # with the current block?
    # A t-subset s is covered by adding x to block iff s contains x and s \ {x} is a (t-1)-subset of block
    # Actually: s is covered iff s ⊆ block ∪ {x} and x ∈ s. So s must contain x, and s\{x} ⊆ block.
    
    while len(block) < k:
        best_x = -1
        best_gain = -1
        for x in range(v):
            if x in block:
                continue
            # Count uncovered t-subsets that contain x and whose other t-1 elements are in block
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                s = tuple(sorted(rest + (x,)))
                if s in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x == -1:
            break
        block.append(best_x)
    
    block.sort()
    return block


def _greedy_cover(v, k, t, deadline, rng):
    """Run greedy covering. Returns list of blocks."""
    all_tsubs, tsub_idx = _precompute_all_tsubsets(v, t)
    uncovered_set = set(all_tsubs)
    blocks = []
    
    start_time = time.time()
    
    while uncovered_set:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset as seed
        seed = list(min(uncovered_set))
        block = _greedy_block(uncovered_set, seed, v, k, t, all_tsubs, tsub_idx)
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for s in itertools.combinations(block, t):
            uncovered_set.discard(tuple(s))
    
    return blocks, len(uncovered_set)


def _try_improve(blocks, v, k, t, deadline, rng):
    """Try to reduce block count via removal and replacement."""
    all_tsubs, tsub_idx = _precompute_all_tsubsets(v, t)
    
    # Count coverage: for each t-subset, how many blocks cover it
    from collections import Counter
    cover_count = Counter()
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[tuple(s)] += 1
    
    # A block can be removed if all its t-subsets are covered by other blocks
    # i.e., every t-subset in the block has cover_count >= 2
    removable = []
    for i, block in enumerate(blocks):
        can_remove = True
        for s in itertools.combinations(block, t):
            if cover_count[tuple(s)] < 2:
                can_remove = False
                break
        if can_remove:
            removable.append(i)
    
    # Try removing one block at a time, then see if we can cover the gaps
    # This is expensive; just try simple removals
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Recompute cover_count
        cover_count = Counter()
        for block in blocks:
            for s in itertools.combinations(block, t):
                cover_count[tuple(s)] += 1
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            can_remove = True
            for s in itertools.combinations(block, t):
                if cover_count[tuple(s)] < 2:
                    can_remove = False
                    break
            if can_remove:
                # Try removing
                new_blocks = blocks[:i] + blocks[i+1:]
                # Check if still a valid covering
                # Recompute uncovered
                covered = set()
                for b in new_blocks:
                    for s in itertools.combinations(b, t):
                        covered.add(tuple(s))
                if len(covered) == len(all_tsubs):
                    blocks = new_blocks
                    improved = True
                    break
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t). Returns list of blocks."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = random.Random(42)
    
    # Run greedy
    blocks, uncovered_count = _greedy_cover(v, k, t, deadline, rng)
    
    if time.time() > deadline:
        return blocks
    
    # Try local improvement
    blocks = _try_improve(blocks, v, k, t, deadline, rng)
    
    return blocks
# EVOLVE-BLOCK-END
