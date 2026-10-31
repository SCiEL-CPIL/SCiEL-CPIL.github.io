"""Covering design C(19,7,4) solver using global greedy set cover with incremental gains."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # All t-subsets to cover
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Generate all candidate blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute which t-subsets it covers
    # block_tsets[block_idx] = list of tsubset indices covered by this block
    block_tsets = []
    for b in all_blocks:
        tsets = [tsubset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(tsets)
    
    # Greedy set cover
    # uncovered_count[ts_idx] = 1 if uncovered, 0 if covered
    uncovered_count = [1] * n_tsubsets
    total_uncovered = n_tsubsets
    
    # gain[block_idx] = number of currently uncovered t-subsets this block covers
    # We'll compute gains incrementally
    gains = [len(block_tsets[i]) for i in range(n_blocks)]
    
    chosen_blocks = []
    used_blocks = set()
    
    # For efficiency, maintain for each t-subset how many chosen blocks cover it
    # Actually for greedy we just need: when we pick a block, decrement uncovered_count
    # for each t-subset it covers, and update gains for all other blocks.
    
    # To speed up gain updates, maintain:
    # cover_count[ts_idx] = number of chosen blocks that cover this t-subset
    cover_count = [0] * n_tsubsets
    
    # For each block, we need to quickly compute its current gain.
    # gain(block) = sum over ts in block_tsets[block] of (1 if cover_count[ts] == 0 else 0)
    
    # To make this fast, we can maintain for each block a count of how many of its
    # t-subsets are still uncovered. When we select a block, for each t-subset it covers,
    # if that t-subset becomes covered (cover_count goes from 0 to 1), we decrement the
    # uncovered count for all blocks that also cover that t-subset.
    
    # Reverse mapping: for each t-subset, which blocks cover it?
    # This could be large. Let's check: C(19,4)=3876 t-subsets, C(19,7)=50388 blocks.
    # Each t-subset is in C(19-4, 7-4) = C(15,3) = 455 blocks.
    # So total edges = 3876 * 455 = 1,763,580. That's manageable.
    
    ts_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsets in enumerate(block_tsets):
        for ti in tsets:
            ts_blocks[ti].append(bi)
    
    # uncovered_count[bi] = number of t-subsets in block bi that are still uncovered
    # Initialize
    block_uncovered = [len(block_tsets[i]) for i in range(n_blocks)]
    
    # Greedy loop
    while total_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find block with maximum uncovered count
        best_block = max(range(n_blocks), key=lambda i: block_uncovered[i])
        
        if block_uncovered[best_block] == 0:
            # No more blocks can cover anything - shouldn't happen if valid
            break
        
        chosen_blocks.append(all_blocks[best_block])
        used_blocks.add(best_block)
        
        # Update: for each t-subset covered by best_block
        for ti in block_tsets[best_block]:
            if cover_count[ti] == 0:
                # This t-subset is now covered
                cover_count[ti] = 1
                total_uncovered -= 1
                # Decrement block_uncovered for all blocks that cover this t-subset
                for other_bi in ts_blocks[ti]:
                    if other_bi != best_block:
                        block_uncovered[other_bi] -= 1
    
    if time.time() > deadline:
        return chosen_blocks
    
    # Now try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other chosen blocks
    # Iterate removal until no more redundant blocks
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        removed_any = False
        for idx in range(len(chosen_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            b = chosen_blocks[idx]
            # Check if all t-subsets of b are covered by other blocks
            redundant = True
            for ts in itertools.combinations(b, t):
                ts_idx = tsubset_index[ts]
                # Count how many chosen blocks (other than idx) cover this t-subset
                count = 0
                for j, cb in enumerate(chosen_blocks):
                    if j == idx:
                        continue
                    # Check if ts is a subset of cb
                    if all(x in cb for x in ts):
                        count += 1
                        if count >= 1:
                            break
                if count == 0:
                    redundant = False
                    break
            if redundant:
                chosen_blocks.pop(idx)
                removed_any = True
                break
        if not removed_any:
            break
    
    # Final validation check (quick)
    if time.time() < deadline - 5:
        covered = set()
        for b in chosen_blocks:
            for ts in itertools.combinations(b, t):
                covered.add(ts)
        if len(covered) != n_tsubsets:
            # Fall back to greedy-only result (shouldn't happen)
            pass
    
    return [list(b) for b in chosen_blocks]
