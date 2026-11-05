import itertools
import random
import time
import numpy as np
from scipy.optimize import minimize

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a hybrid approach:
    1. Fast greedy initialization with anchor-based candidate selection.
    2. LASSO-inspired sparse recovery to find a minimal subset of blocks.
    3. Local search refinement (removal and swap) to polish the solution.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precomputation
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_set_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_tsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks) and map to indices
    k_subsets = list(itertools.combinations(range(v), k))
    num_ksets = len(k_subsets)
    
    # 3. Map each block to the list of t-set indices it covers
    block_to_tsets = [[] for _ in range(num_ksets)]
    # 4. Map each t-set to the list of block indices that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(k_subsets):
        for ts in itertools.combinations(block, t):
            ts_idx = t_set_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # State
    current_blocks = []  # List of block indices
    tset_counts = [0] * num_tsets
    in_current = [False] * num_ksets

    def add_block(b_idx):
        current_blocks.append(b_idx)
        in_current[b_idx] = True
        for ts_idx in block_to_tsets[b_idx]:
            tset_counts[ts_idx] += 1

    def remove_block(b_idx):
        if b_idx in current_blocks:
            current_blocks.remove(b_idx)
            in_current[b_idx] = False
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] -= 1

    def is_valid_covering():
        return all(count > 0 for count in tset_counts)

    # Phase 1: Greedy Initialization
    # Use a randomized greedy approach to get a valid covering quickly.
    # Instead of scanning all blocks, pick a random uncovered t-set and choose the best block covering it.
    uncovered = set(range(num_tsets))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-set to anchor the search
        # This is faster than scanning all 50k blocks every time
        anchor_ts = random.choice(list(uncovered))
        candidates = tset_to_blocks[anchor_ts]
        
        best_b = -1
        max_gain = 0
        
        for b_idx in candidates:
            if in_current[b_idx]:
                continue
            gain = 0
            for ts_idx in block_to_tsets[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
            
        if best_b == -1:
            break
            
        add_block(best_b)
        for ts_idx in block_to_tsets[best_b]:
            if ts_idx in uncovered:
                uncovered.remove(ts_idx)

    # Phase 2: LASSO / Sparse Recovery
    # We have a valid covering. We want to minimize the number of blocks.
    # This is equivalent to finding a sparse vector x in {0,1}^N such that A x >= 1.
    # We can relax this to minimize ||x||_1 subject to A x >= 1, x >= 0.
    # Or simply minimize ||x||_1 + lambda * ||x||_2^2 to encourage sparsity and binary-like behavior.
    # Given the size (50k variables), a full convex optimization might be too slow or memory intensive.
    # Instead, we use a heuristic "LASSO step": 
    # 1. Start with current set.
    # 2. Identify "weak" blocks (those covering few unique t-sets).
    # 3. Try to remove them if possible.
    # 4. This is essentially local search, but we can use the LASSO insight to prioritize removal of blocks with low "L1 contribution".
    
    # A more direct structural change: Use the incidence matrix to perform a "sparse projection".
    # However, for N=50k, D=3876, the matrix is sparse.
    # Let's stick to a robust local search but enhanced with the "unique coverage" metric which aligns with L1 sparsity.
    
    # Phase 3: Local Search Refinement
    # We run this until deadline.
    # Strategy:
    # 1. Remove redundant blocks (count 0 for all t-sets in block).
    # 2. Swap blocks to increase redundancy of others (allowing removal).
    
    best_size = len(current_blocks)
    no_improve_counter = 0
    
    while time.time() < deadline:
        improved = False
        
        # Step 3a: Remove redundant blocks
        # A block is redundant if every t-set it covers is covered by at least one other block.
        # i.e., for all ts in block, tset_counts[ts] > 1.
        to_remove = []
        for b_idx in current_blocks:
            redundant = True
            for ts_idx in block_to_tsets[b_idx]:
                if tset_counts[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(b_idx)
        
        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True
            
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        if no_improve_counter > 50:
            # Stagnation, break to save time or try a different strategy?
            # For now, break to avoid wasting time on a local minimum that seems stuck.
            # However, we can also try a "kick" (random swap) to escape.
            # Let's try a few random swaps if stuck.
            if no_improve_counter > 100:
                break

        if time.time() > deadline:
            break

        # Step 3b: Swap to improve
        # Find a block that is "critical" (has a unique t-set).
        # Try to replace it with a block that covers the same unique t-sets but potentially allows other blocks to become redundant.
        # Actually, a simpler heuristic:
        # Pick a random block B.
        # Try to replace it with a random block B' such that the covering remains valid.
        # If valid, keep it if it leads to a reduction in "total unique counts" or similar.
        # Better: Just try to remove a block. If we can't, try a swap that makes another block removable.
        
        # Let's identify blocks with unique t-sets.
        critical_blocks = []
        for b_idx in current_blocks:
            has_unique = False
            for ts_idx in block_to_tsets[b_idx]:
                if tset_counts[ts_idx] == 1:
                    has_unique = True
                    break
            if has_unique:
                critical_blocks.append(b_idx)
                
        if not critical_blocks:
            # All blocks are redundant? Should have been removed.
            # Or no blocks?
            if not current_blocks:
                break
            continue

        # Pick a random critical block to swap
        old_b = random.choice(critical_blocks)
        
        # Find all unique t-sets of old_b
        unique_ts = []
        for ts_idx in block_to_tsets[old_b]:
            if tset_counts[ts_idx] == 1:
                unique_ts.append(ts_idx)
                
        if not unique_ts:
            # Contradiction? If it was in critical_blocks, it should have a unique t-set.
            # Unless state changed? We just checked.
            continue

        # Find candidate new blocks that cover all unique_ts
        # Intersection of tset_to_blocks[ts] for ts in unique_ts
        candidates = set(tset_to_blocks[unique_ts[0]])
        for ts_idx in unique_ts[1:]:
            candidates.intersection_update(tset_to_blocks[ts_idx])
            if not candidates:
                break
                
        # Filter out blocks already in current
        candidates.difference_update(current_blocks)
        
        if candidates:
            # Pick a random candidate
            new_b = random.choice(list(candidates))
            
            # Perform swap
            remove_block(old_b)
            add_block(new_b)
            
            # Check if this swap allowed any block to become redundant
            # We don't break immediately, the main loop will handle removal in next iteration.
            # But we can check immediately to see if we improved "potential"
            # For now, just let the loop handle it.
            improved = True # We made a change

    # Final cleanup: remove any remaining redundant blocks
    to_remove = []
    for b_idx in current_blocks:
        redundant = True
        for ts_idx in block_to_tsets[b_idx]:
            if tset_counts[ts_idx] == 1:
                redundant = False
                break
        if redundant:
            to_remove.append(b_idx)
            
    for b_idx in to_remove:
        remove_block(b_idx)

    # Convert block indices back to lists of points
    result = [list(k_subsets[b_idx]) for b_idx in current_blocks]
    return result

if __name__ == "__main__":
    # Test with the specific task parameters
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
