"""Covering design C(19,7,4) via greedy set cover with local search (block swaps)."""
import itertools
import random
import time
from collections import defaultdict
import heapq
import numpy as np

# We will use a deadline guard.
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # For each k-subset (block), precompute the list of t-subsets it covers
    # But there are C(19,7) = 50388 blocks, each covering C(7,4)=35 t-subsets.
    # That's manageable.
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubsets = len(all_ksubsets)

    # For each block, compute the set of t-subset indices it covers
    # Using a set of indices for fast operations
    block_covers = []
    for block in all_ksubsets:
        covered = set()
        for ts in itertools.combinations(block, t):
            covered.add(tsubset_index[ts])
        block_covers.append(covered)

    # Precompute for each t-subset which blocks cover it
    tsubset_to_blocks = defaultdict(list)
    for bi in range(n_ksubsets):
        for ts_idx in block_covers[bi]:
            tsubset_to_blocks[ts_idx].append(bi)
    
    # Convert to sets for faster intersection operations
    tsubset_to_block_sets = {ts_idx: set(lst) for ts_idx, lst in tsubset_to_blocks.items()}

    def greedy_set_cover(seed):
        """
        Greedy set cover using a max-heap for efficiency.
        Returns list of block indices.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_tsubsets))
        selected_blocks = []
        
        # gain[bi] = number of uncovered t-subsets covered by block bi
        gain = [0] * n_ksubsets
        for bi in range(n_ksubsets):
            gain[bi] = len(block_covers[bi] & uncovered)
        
        # Max-heap: (-gain, bi)
        heap = [(-gain[bi], bi) for bi in range(n_ksubsets)]
        heapq.heapify(heap)
        
        while uncovered and time.time() < deadline - 10:
            best_bi = -1
            max_gain = 0
            
            # Pop until we find a valid (non-stale) entry with positive gain
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                # Check if this gain is still valid (matches current gain array)
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
                # Otherwise, it's a stale entry, discard
            
            if max_gain <= 0:
                break
                
            selected_blocks.append(best_bi)
            
            # Update gains for all blocks that share t-subsets with best_bi
            for ts_idx in block_covers[best_bi]:
                if ts_idx in uncovered:
                    uncovered.remove(ts_idx)
                    # Decrement gain for all blocks covering this ts_idx
                    for bj in tsubset_to_blocks[ts_idx]:
                        gain[bj] -= 1
                        # Push updated gain into heap
                        heapq.heappush(heap, (-gain[bj], bj))
                        
            # Mark best_bi as used (gain -1 to prevent re-selection)
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
        return selected_blocks

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
            for ts_idx in block_covers[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in block_set:
                redundant = True
                for ts_idx in block_covers[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
                    
            if to_remove:
                for bi in to_remove:
                    block_set.remove(bi)
                    for ts_idx in block_covers[bi]:
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
                for ts_idx in block_covers[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # This block is redundant (should have been caught in Phase 1, but just in case)
                    block_set.remove(bi)
                    for ts_idx in block_covers[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart local search loop
                    
                # Find candidate blocks that cover ALL unique_tsubs
                # Start with the rarest unique tsubset to minimize candidates
                min_ts = min(unique_tsubs, key=lambda x: len(tsubset_to_blocks[x]))
                candidates = set(tsubset_to_block_sets[min_ts])
                
                for ts_idx in unique_tsubs[1:]:
                    if not candidates:
                        break
                    candidates &= tsubset_to_block_sets[ts_idx]
                    
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
                    for ts_idx in block_covers[bi]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(bj)
                    for ts_idx in block_covers[bj]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop after a swap
                    
                if swapped:
                    break
            
            # Phase 3: Try 2-for-1 merges
            if not improved and time.time() < deadline - 5:
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                
                for i in range(len(current_blocks)):
                    if time.time() > deadline - 5:
                        break
                    bi = current_blocks[i]
                    if bi not in block_set:
                        continue
                        
                    for j in range(i + 1, len(current_blocks)):
                        bj = current_blocks[j]
                        if bj not in block_set:
                            continue
                            
                        union_ts = block_covers[bi] | block_covers[bj]
                        
                        # Find a block that covers all t-subsets in union_ts
                        # Try blocks that cover at least one t-subset from each block
                        candidate_blocks = set(tsubset_to_block_sets[next(iter(union_ts))])
                        
                        for bk in candidate_blocks:
                            if bk in block_set:
                                continue
                            if union_ts.issubset(block_covers[bk]):
                                # Replace blocks i and j with block bk
                                block_set.remove(bi)
                                block_set.remove(bj)
                                for ts_idx in block_covers[bi]:
                                    cover_count[ts_idx] -= 1
                                for ts_idx in block_covers[bj]:
                                    cover_count[ts_idx] -= 1
                                
                                block_set.add(bk)
                                for ts_idx in block_covers[bk]:
                                    cover_count[ts_idx] += 1
                                
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return list(block_set)

    # Run multiple greedy passes with different seeds and keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts to try
    n_restarts = 10
    
    for seed in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        greedy_result = greedy_set_cover(seed)
        
        if not greedy_result:
            continue
            
        # Apply local search
        refined = local_search(greedy_result, seed)
        
        # Verify coverage
        covered = set()
        for bi in refined:
            covered |= block_covers[bi]
            
        if covered == set(range(n_tsubsets)):
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined
                # If we hit the known lower bound or better, stop early
                if best_len <= 131:
                    break
    
    # If no valid solution was found, fall back to a simple greedy
    if best_blocks is None:
        greedy_result = greedy_set_cover(0)
        # Ensure all t-subsets are covered
        covered = set()
        for bi in greedy_result:
            covered |= block_covers[bi]
        
        if covered != set(range(n_tsubsets)):
            # Add more blocks to cover remaining t-subsets
            uncovered = set(range(n_tsubsets)) - covered
            while uncovered:
                best_bi = -1
                best_count = -1
                for bi in range(n_ksubsets):
                    if bi in greedy_result:
                        continue
                    cov = len(block_covers[bi] & uncovered)
                    if cov > best_count:
                        best_count = cov
                        best_bi = bi
                if best_bi == -1:
                    break
                greedy_result.append(best_bi)
                covered |= block_covers[best_bi]
                uncovered -= block_covers[best_bi]
            best_blocks = greedy_result
        else:
            best_blocks = greedy_result
    
    # Convert indices back to actual blocks
    blocks = [list(all_ksubsets[bi]) for bi in best_blocks]
    
    return blocks
