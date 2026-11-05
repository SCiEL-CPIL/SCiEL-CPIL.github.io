"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter

import numpy as np


def _all_blocks(v, k):
    return list(itertools.combinations(range(v), k))


def _block_index_to_tuple(idx, v, k):
    """Convert a block index (lexicographic) to a tuple of k points."""
    # Precompute combinations for small v,k? For v=19,k=7, C(19,7)=50388.
    # We can just use itertools.combinations and cache or generate on the fly.
    # Since we need to map index -> block often, let's precompute a list of blocks.
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all blocks and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each t-subset to a canonical index
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Build coverage matrix: cov[block_idx][tsubset_idx] = 1 if tsubset is in block
    # This is a large sparse structure. For v=19,k=7,t=4:
    # num_blocks = 50388, num_tsubsets = 3876
    # Each block covers C(7,4)=35 t-subsets.
    # Total non-zeros: 50388 * 35 = 1,763,580. Manageable.
    
    # We'll use a list of sets: for each block, which t-subsets it covers
    block_covers = [set() for _ in range(num_blocks)]
    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            t_idx = tsubset_to_idx[ts]
            block_covers[b_idx].add(t_idx)
    
    # Greedy set cover
    uncovered = set(range(num_tsubsets))
    blocks_used = []
    
    while uncovered:
        # Find block covering most uncovered t-subsets
        best_block = -1
        best_count = -1
        for b_idx in range(num_blocks):
            count = len(block_covers[b_idx] & uncovered)
            if count > best_count:
                best_count = count
                best_block = b_idx
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        if best_count <= 0:
            break  # Should not happen if covering exists
        
        blocks_used.append(all_blocks[best_block])
        uncovered.difference_update(block_covers[best_block])
    
    if time.time() > deadline:
        return [list(b) for b in blocks_used]
    
    # Now try to improve with local search / repair
    # Represent solution as a set of block indices
    solution = set(blocks_used)
    
    # Compute coverage counts for each t-subset
    coverage_count = np.zeros(num_tsubsets, dtype=np.int32)
    for b_idx in solution:
        for t_idx in block_covers[b_idx]:
            coverage_count[t_idx] += 1
    
    # Try to remove blocks that don't make any t-subset uncovered
    # A block is redundant if removing it doesn't uncover any t-subset
    # i.e., for all t_idx in block_covers[b_idx], coverage_count[t_idx] > 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for b_idx in list(solution):
            if time.time() > deadline:
                break
            # Check if removing b_idx is safe
            redundant = True
            for t_idx in block_covers[b_idx]:
                if coverage_count[t_idx] == 1:
                    redundant = False
                    break
            if redundant:
                solution.discard(b_idx)
                for t_idx in block_covers[b_idx]:
                    coverage_count[t_idx] -= 1
                improved = True
    
    # Now try to swap blocks: replace one block with another that covers
    # more uncovered t-subsets (after removal) or same coverage with fewer total
    # Actually, let's do a different improvement: try to find a block that
    # covers t-subsets that are currently covered by exactly one block,
    # and see if we can swap.
    
    # Better approach: randomized local search
    # Try to find a smaller covering by random restarts of greedy with different orderings
    
    # For now, let's try a simple improvement: 
    # For each block in solution, try to replace it with a block that covers
    # all the same t-subsets that are uniquely covered by the original block.
    
    # Actually, let's just do multiple greedy passes with random tie-breaking
    # and keep the best result.
    
    best_solution = set(solution)
    best_size = len(best_solution)
    
    # Try several randomized greedy runs
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Randomize the order of blocks for tie-breaking
        uncovered = set(range(num_tsubsets))
        current_solution = set()
        
        # Precompute block_covers as frozensets for faster intersection
        # Actually, let's just do greedy with random tie-breaking
        while uncovered:
            # Find all blocks with max coverage
            best_count = -1
            candidates = []
            for b_idx in range(num_blocks):
                if b_idx in current_solution:
                    continue
                count = len(block_covers[b_idx] & uncovered)
                if count > best_count:
                    best_count = count
                    candidates = [b_idx]
                elif count == best_count and count > 0:
                    candidates.append(b_idx)
            
            if not candidates:
                break
            
            # Randomly pick from candidates
            chosen = rng.choice(candidates)
            current_solution.add(chosen)
            uncovered.difference_update(block_covers[chosen])
        
        if len(current_solution) < best_size and time.time() < deadline:
            best_solution = current_solution
            best_size = len(best_solution)
    
    # Convert back to list of blocks
    result = [list(all_blocks[b_idx]) for b_idx in best_solution]
    result.sort()
    return result

# EVOLVE-BLOCK-END
