"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def _greedy_cover_vectorized(v, k, t, deadline):
    """
    Fast greedy set cover using numpy for the coverage matrix.
    Returns list of blocks.
    """
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build block->triple incidence: for each block, which triple indices it covers
    # Use a sparse-like representation: list of arrays of triple indices per block
    block_triple_indices = []
    for block in all_blocks:
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[triple])
        block_triple_indices.append(indices)
    
    # Coverage count per triple (how many blocks cover it)
    triple_cover_count = np.zeros(num_triples, dtype=np.int32)
    # For each block, count of uncovered triples it covers
    block_uncovered = np.zeros(num_blocks, dtype=np.int32)
    
    # Initialize: each block covers C(k,t) triples
    for bi in range(num_blocks):
        for ti in block_triple_indices[bi]:
            triple_cover_count[ti] += 1
        block_uncovered[bi] = len(block_triple_indices[bi])
    
    # Uncovered mask
    uncovered_mask = np.ones(num_triples, dtype=bool)
    num_uncovered = num_triples
    
    blocks = []
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find best block
        best_block = int(np.argmax(block_uncovered))
        best_count = block_uncovered[best_block]
        if best_count <= 0:
            break
        
        blocks.append(list(all_blocks[best_block]))
        
        # Update coverage
        for ti in block_triple_indices[best_block]:
            if uncovered_mask[ti]:
                uncovered_mask[ti] = False
                num_uncovered -= 1
                # Decrement block_uncovered for all blocks covering this triple
                # We need to know which blocks cover triple ti
                # Precompute triple_to_blocks for this
                pass
    
    return blocks


def _greedy_cover_with_tracking(v, k, t, deadline):
    """
    Greedy set cover with full tracking for post-processing.
    Returns (blocks, triple_to_block_ids) where triple_to_block_ids maps
    each triple to the list of block indices that cover it.
    """
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, the triple indices it covers
    block_triple_indices = []
    for block in all_blocks:
        indices = []
        for triple in itertools.combinations(block, t):
            indices.append(triple_to_idx[triple])
        block_triple_indices.append(indices)
    
    # triple_to_blocks: for each triple, list of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi in range(num_blocks):
        for ti in block_triple_indices[bi]:
            triple_to_blocks[ti].append(bi)
    
    # block_uncovered_count[bi] = number of currently uncovered triples in block bi
    block_uncovered_count = np.array([len(bti) for bti in block_triple_indices], dtype=np.int32)
    
    # uncovered: boolean array
    uncovered = np.ones(num_triples, dtype=bool)
    num_uncovered = num_triples
    
    blocks = []
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        best_block = int(np.argmax(block_uncovered_count))
        best_count = block_uncovered_count[best_block]
        if best_count <= 0:
            break
        
        blocks.append(list(all_blocks[best_block]))
        
        # Update
        for ti in block_triple_indices[best_block]:
            if uncovered[ti]:
                uncovered[ti] = False
                num_uncovered -= 1
                for bi2 in triple_to_blocks[ti]:
                    block_uncovered_count[bi2] -= 1
    
    return blocks, triple_to_blocks, block_triple_indices


def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks: if a block's t-subsets are all covered by other blocks,
    remove it.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # For each block, compute which triples it covers
    block_triples = []
    for block in blocks:
        triples = set()
        for triple in itertools.combinations(block, t):
            triples.add(triple_to_idx[triple])
        block_triples.append(triples)
    
    # For each triple, how many blocks cover it
    triple_count = np.zeros(num_triples, dtype=np.int32)
    for bt in block_triples:
        for ti in bt:
            triple_count[ti] += 1
    
    # Try to remove blocks in order of lowest "uniqueness" (fewest unique triples)
    # A block is redundant if all its triples are covered by other blocks
    # i.e., for every triple in the block, triple_count[ti] >= 2
    
    # Sort blocks by number of unique triples (triples with count==1)
    block_unique_counts = []
    for bt in block_triples:
        unique = sum(1 for ti in bt if triple_count[ti] == 1)
        block_unique_counts.append(unique)
    
    # Try removing blocks with 0 unique triples first
    order = sorted(range(n), key=lambda i: block_unique_counts[i])
    
    kept = list(range(n))
    kept_set = set(kept)
    
    for bi in order:
        if time.time() > deadline:
            break
        if bi not in kept_set:
            continue
        # Check if block bi is redundant given current kept set
        # Recompute: for each triple in block bi, is it covered by another kept block?
        redundant = True
        for ti in block_triples[bi]:
            # Count how many kept blocks cover ti
            # We need an efficient way... 
            # For now, just check triple_count but that includes bi itself
            # So we need: is there another kept block covering ti?
            # This is expensive to do naively. Let's use a different approach.
            pass
        # Simplified: if triple_count[ti] >= 2 for all ti in block, and we haven't
        # removed other blocks that were the only other coverers, it might still be safe.
        # But to be safe, let's do a proper check.
    
    # Simpler approach: iteratively try to remove each block and verify
    # This is O(n * num_triples_per_block * n) which is too slow for large n.
    # Let's use a batch approach: remove all blocks with 0 unique triples simultaneously,
    # then recheck.
    
    # Actually, let's just do a simple pass: remove blocks where all their triples
    # have count >= 2 (in the current set), then recompute counts.
    # Repeat until no more removals.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Recompute triple counts
        triple_count = np.zeros(num_triples, dtype=np.int32)
        for bi in kept:
            for ti in block_triples[bi]:
                triple_count[ti] += 1
        
        new_kept = []
        for bi in kept:
            # Check if bi is redundant
            is_redundant = True
            for ti in block_triples[bi]:
                if triple_count[ti] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                changed = True
            else:
                new_kept.append(bi)
        
        kept = new_kept
    
    return [blocks[i] for i in kept]


def _local_search(blocks, v, k, t, deadline, max_iters=50):
    """
    Local search: try replacing a block with another that maintains coverage
    but might allow further reduction. Actually, for minimizing block count,
    we try to remove blocks. This is similar to redundancy removal but more aggressive.
    
    Alternative: try swapping points in a block to see if it still covers the same
    unique triples, potentially allowing another block to be removed.
    
    For now, let's just do iterative redundancy removal with randomization.
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    n = len(blocks)
    if n == 0:
        return blocks
    
    # Precompute block triples
    block_triples = []
    for block in blocks:
        triples = set()
        for triple in itertools.combinations(block, t):
            triples.add(triple_to_idx[triple])
        block_triples.append(triples)
    
    # Try to remove blocks one at a time, checking coverage
    # For efficiency, maintain triple_cover_count
    triple_cover_count = np.zeros(num_triples, dtype=np.int32)
    for bt in block_triples:
        for ti in bt:
            triple_cover_count[ti] += 1
    
    kept = list(range(n))
    kept_set = set(kept)
    
    # Shuffle to break ties
    order = list(range(n))
    random.shuffle(order)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        made_change = False
        for bi in order:
            if time.time() > deadline:
                break
            if bi not in kept_set:
                continue
            # Check if removing bi leaves all triples covered
            can_remove = True
            for ti in block_triples[bi]:
                if triple_cover_count[ti] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove bi
                for ti in block_triples[bi]:
                    triple_cover_count[ti] -= 1
                kept_set.discard(bi)
                made_change = True
        if not made_change:
            break
    
    return [blocks[i] for i in kept]


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1)
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Phase 1: Greedy set cover with tracking
    blocks, triple_to_blocks, block_triple_indices = _greedy_cover_with_tracking(v, k, t, deadline)
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant(blocks, v, k, t, deadline)
    
    # Phase 3: Local search for further reduction
    blocks = _local_search(blocks, v, k, t, deadline)
    
    return blocks

# EVOLVE-BLOCK-END
