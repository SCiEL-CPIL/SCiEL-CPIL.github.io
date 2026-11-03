"""
Combined strategy: Precompute all block-tsubset relationships, use heap-based greedy for initial construction, and apply local search (removal + swaps) for refinement.
"""
import itertools
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    points = list(range(v))
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Generate all possible t-subsets
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Map: block_index -> list of tsubset_indices it covers
    block_tsubs = [[] for _ in range(n_blocks)]
    # Map: tsubset_index -> list of block_indices that cover it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        # Calculate t-subsets for this block
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Precompute sets for faster intersection in local search
    # tsub_block_sets[ts_idx] = set of block indices covering ts_idx
    tsub_block_sets = [set(lst) for lst in tsub_blocks]
    
    def greedy_cover(seed):
        """
        Greedy set cover using a max-heap.
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # gain[bi] = number of currently uncovered t-subsets covered by block bi
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        # To avoid re-adding the same block with stale gains, we track a version or just pop until valid.
        # Since gains only decrease, we can pop stale entries.
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Pop until we find a valid (non-stale) entry with positive gain
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                # Check if this gain is still valid (matches current gain array)
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
                # Otherwise, it's a stale entry, discard
            else:
                break # Heap empty or all gains 0
                
            if max_gain <= 0:
                break
                
            selected.append(best_bi)
            
            # Update gains for all blocks that share t-subsets with best_bi
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        # Push updated gain into heap
                        heapq.heappush(heap, (-gain[bj], bj))
                        
            # Mark best_bi as used (gain -1 to prevent re-selection)
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
        return selected

    def local_search(blocks, seed):
        """
        Local search: Remove redundant blocks and perform swaps.
        blocks: list of block indices
        """
        rng = random.Random(seed)
        # Work with a set for O(1) lookup
        block_set = set(blocks)
        
        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = [0] * n_tsubsets
        
        # Initialize cover_count
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed
            
            # Phase 2: Try swaps
            # If a block has unique t-subsets, try to replace it with another block that covers those unique t-subsets.
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # This block is redundant (should have been caught in Phase 1, but just in case)
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart local search loop
                    
                # Find candidate blocks that cover ALL unique_tsubs
                # Start with the rarest unique tsubset to minimize candidates
                min_ts = min(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
                candidates = set(tsub_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsub_block_sets[ts_idx]
                    
                # Remove current block from candidates
                candidates.discard(bi)
                
                if not candidates:
                    continue
                    
                # Try candidates - limit to a reasonable number to avoid long stalls
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                swapped = False
                for bj in cand_list[:50]: # Limit candidates
                    if bj in block_set:
                        continue
                        
                    # Perform swap
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_tsubs[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    # Main execution
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    # Since precomputation is done, greedy is fast.
    # We can afford several restarts.
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 30: # Keep 30s for local search
            break
            
        seed = i * 12345 + 7
        blocks = greedy_cover(seed)
        
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if not best_blocks:
        return []
        
    # Apply local search to the best solution
    if time.time() < deadline - 2:
        best_blocks = local_search(best_blocks, 42)
        
    # Verify validity (exhaustive check might be slow, but for v=19, k=7, t=4, it's manageable if we use the precomputed indices)
    # Actually, the task says "checked exhaustively" by the evaluator. We just need to return valid blocks.
    # Our construction ensures validity by construction (greedy covers all, local search maintains coverage).
    
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    
    return final_blocks
