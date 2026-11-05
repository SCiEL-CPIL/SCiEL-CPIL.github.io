"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using Greedy Set Cover with Bitsets
    and Local Search (Redundancy Removal + Block Swapping).
    """
    # 1. Generate all triples and map to indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # 2. Generate all blocks and their bitsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute bitset for each block
    # A bitset is a Python int where bit i is 1 if triple i is covered by this block
    block_bitsets = []
    # Also pre-compute triple_to_blocks for local search
    triple_to_blocks = defaultdict(list)
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            triple_to_blocks[idx].append(i)
        block_bitsets.append(mask)
    
    # 3. Greedy Set Cover
    full_mask = (1 << num_triples) - 1
    uncovered_mask = full_mask
    selected_block_indices = []
    used_block_indices = set()
    
    # Optimization: Pre-compute list of block indices
    block_indices = list(range(num_blocks))
    
    for _ in range(num_triples + 10):
        if uncovered_mask == 0:
            break
        
        best_block_idx = -1
        best_gain = 0
        
        for i in block_indices:
            if i in used_block_indices:
                continue
            
            common = uncovered_mask & block_bitsets[i]
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        selected_block_indices.append(best_block_idx)
        used_block_indices.add(best_block_idx)
        uncovered_mask &= ~block_bitsets[best_block_idx]
        
        if best_gain == 0:
            break

    # 4. Local Search Phase
    # We operate on the list of selected block indices
    current_blocks = list(selected_block_indices)
    
    # Build coverage counts
    cover_count = np.zeros(num_triples, dtype=np.int32)
    for b_idx in current_blocks:
        for triple in all_blocks[b_idx]:
            t_idx = triple_to_idx[triple]
            cover_count[t_idx] += 1

    # Helper to update cover count when adding/removing a block
    def update_cover_count(b_idx, delta):
        for triple in all_blocks[b_idx]:
            t_idx = triple_to_idx[triple]
            cover_count[t_idx] += delta

    # Local Search Loop
    max_iterations = 100
    for _ in range(max_iterations):
        improved = False
        # Shuffle order to avoid bias
        random.shuffle(current_blocks)
        
        for i in range(len(current_blocks)):
            b_idx = current_blocks[i]
            
            # 1. Check for Redundancy
            is_redundant = True
            unique_triples = []
            for triple in all_blocks[b_idx]:
                t_idx = triple_to_idx[triple]
                if cover_count[t_idx] <= 1:
                    is_redundant = False
                    unique_triples.append(t_idx)
                    break
            
            if is_redundant:
                # Remove block
                update_cover_count(b_idx, -1)
                current_blocks.pop(i)
                improved = True
                break # Restart loop with new state
            
            # 2. Check for Swap
            # If not redundant, unique_triples contains at least one triple
            # We need to find a block B_new not in current_blocks that covers all unique_triples
            # Note: unique_triples might be incomplete if we broke early.
            # We need the FULL list of unique triples to ensure B_new covers ALL of them.
            unique_triples_full = []
            for triple in all_blocks[b_idx]:
                t_idx = triple_to_idx[triple]
                if cover_count[t_idx] == 1:
                    unique_triples_full.append(t_idx)
            
            if not unique_triples_full:
                # Should have been caught by redundancy check
                continue

            # Find candidate blocks
            # Intersection of triple_to_blocks for each unique triple
            candidates = None
            for t_idx in unique_triples_full:
                blocks_covering_t = triple_to_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_covering_t)
                else:
                    candidates.intersection_update(blocks_covering_t)
            
            if candidates:
                # Filter out blocks already in current_blocks
                current_set = set(current_blocks)
                valid_candidates = candidates - current_set
                
                if valid_candidates:
                    # Pick a candidate. 
                    # Heuristic: Pick the one that covers the most other triples (potential for future removals)
                    # Or just pick the first one.
                    # Let's pick the one with max gain on currently covered triples?
                    # Actually, any valid swap maintains coverage.
                    # To encourage future removals, pick a block that overlaps more with others?
                    # Let's just pick the first one for speed.
                    new_b_idx = next(iter(valid_candidates))
                    
                    # Perform Swap
                    update_cover_count(b_idx, -1)
                    update_cover_count(new_b_idx, 1)
                    current_blocks[i] = new_b_idx
                    improved = True
                    break # Restart loop
        
        if not improved:
            break

    # 5. Convert back to list of blocks
    return [list(all_blocks[idx]) for idx in current_blocks]

# EVOLVE-BLOCK-END
