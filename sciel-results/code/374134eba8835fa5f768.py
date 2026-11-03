import itertools
import time
import random
import numpy as np
from collections import defaultdict
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # block_tsubs[bi] = list of tsubset indices covered by block bi
    block_tsubs = [None] * n_blocks
    # tsub_blocks[ts_idx] = list of block indices covering tsubset ts_idx
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Precompute sets for block_tsubs for O(1) membership checks in swap logic
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    
    # Precompute lengths of tsub_blocks for "rarest" selection
    tsub_block_lens = [len(lst) for lst in tsub_blocks]

    def greedy_cover(seed):
        """
        Greedy set cover using a max-heap.
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        gain = [0] * n_blocks
        for bi in range(n_blocks):
            gain[bi] = len(block_tsubs[bi])
            
        heap = [(-gain[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        
        selected = []
        
        while True:
            if time.time() > deadline - 10:
                break
                
            best_bi = -1
            max_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if g == gain[bi] and g > 0:
                    best_bi = bi
                    max_gain = g
                    break
            else:
                break
                
            if max_gain <= 0:
                break
                
            selected.append(best_bi)
            
            for ts_idx in block_tsubs[best_bi]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for bj in tsub_blocks[ts_idx]:
                        gain[bj] -= 1
                        heapq.heappush(heap, (-gain[bj], bj))
                        
            gain[best_bi] = -1
            heapq.heappush(heap, (-gain[best_bi], best_bi))
            
        return selected

    def local_search(blocks, seed):
        """
        Optimized Local Search with Iterated Perturbation.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        
        # cover_count[ts_idx] = number of blocks in block_set covering ts_idx
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        # We run multiple iterations of (Remove -> Swap -> Perturb)
        # We limit the number of perturbations to avoid getting stuck in bad cycles
        max_perturbations = 50
        perturbations_used = 0
        
        while time.time() < deadline - 5:
            improved = False
            
            # --- Phase 1: Remove Redundant Blocks ---
            # A block is redundant if all its t-subsets are covered by at least one other block.
            # i.e., cover_count[ts] >= 2 for all ts in block_tsubs[bi]
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
                # After removal, we should restart the swap phase as structure changed
                # But we can continue to swap phase immediately
                continue 

            # --- Phase 2: 1-for-1 Swaps ---
            # For each block, if it has unique t-subsets, try to find a replacement
            # that covers those unique t-subsets.
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > deadline - 5:
                    break
                if bi not in block_set:
                    continue
                    
                # Find unique t-subsets for bi
                unique_tsubs = []
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been removed in Phase 1, but handle just in case
                    block_set.remove(bi)
                    for ts_idx in block_tsubs[bi]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart loop
                    
                # Find the rarest unique t-subset to minimize candidates
                min_ts = min(unique_tsubs, key=lambda x: tsub_block_lens[x])
                
                # Iterate through blocks covering min_ts
                # We check if they cover ALL unique_tsubs
                found_swap = False
                for bj in tsub_blocks[min_ts]:
                    if bj == bi or bj in block_set:
                        continue
                    
                    # Check if bj covers all unique_tsubs
                    # Using set membership is fast
                    if all(ts in block_tsub_sets[bj] for ts in unique_tsubs):
                        # Perform swap
                        block_set.remove(bi)
                        for ts_idx in block_tsubs[bi]:
                            cover_count[ts_idx] -= 1
                            
                        block_set.add(bj)
                        for ts_idx in block_tsubs[bj]:
                            cover_count[ts_idx] += 1
                            
                        improved = True
                        found_swap = True
                        break # Restart local search loop
                        
                if found_swap:
                    break # Restart while loop
            
            if improved:
                continue # Restart while loop to check for new redundancies/swaps
            
            # --- Phase 3: Perturbation (Iterated Local Search) ---
            # If no improvement found, we are likely in a local minimum.
            # Perturb the solution to escape.
            if perturbations_used < max_perturbations:
                perturbations_used += 1
                # Randomly swap a few blocks
                # Pick 2-5 random blocks and try to swap them with random alternatives
                n_perturb = rng.randint(2, 5)
                current_blocks = list(block_set)
                rng.shuffle(current_blocks)
                
                for _ in range(n_perturb):
                    if not current_blocks:
                        break
                    bi = current_blocks.pop()
                    if bi not in block_set:
                        continue
                    
                    # Try to swap bi with a random block not in set
                    # To ensure validity, we pick a random block and check if it covers
                    # the unique t-subsets of bi. If not, we just pick a random block
                    # and accept the "damage" (uncovered t-subsets) if we can't find a valid swap?
                    # No, we must maintain a valid cover.
                    # So we only perturb if we can find a valid swap.
                    # If we can't find a valid swap for bi, we skip this perturbation step for bi.
                    
                    unique_tsubs = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                    if not unique_tsubs:
                        # Remove it
                        block_set.remove(bi)
                        for ts_idx in block_tsubs[bi]:
                            cover_count[ts_idx] -= 1
                        continue
                    
                    min_ts = min(unique_tsubs, key=lambda x: tsub_block_lens[x])
                    candidates = [bj for bj in tsub_blocks[min_ts] if bj not in block_set and bj != bi]
                    
                    if not candidates:
                        continue
                    
                    # Pick a random candidate
                    bj = rng.choice(candidates)
                    
                    # Verify it covers all unique tsubs (it covers min_ts by definition, but must cover others)
                    if all(ts in block_tsub_sets[bj] for ts in unique_tsubs):
                        block_set.remove(bi)
                        for ts_idx in block_tsubs[bi]:
                            cover_count[ts_idx] -= 1
                        block_set.add(bj)
                        for ts_idx in block_tsubs[bj]:
                            cover_count[ts_idx] += 1
                    else:
                        # If the random candidate doesn't work, try a few more
                        # Or just skip this perturbation
                        pass
            
            # If we've used all perturbations and made no improvements, we might be stuck.
            # In a real ILS, we might restart from scratch or use a different heuristic.
            # For now, we break if we've exhausted perturbations and no improvement.
            if perturbations_used >= max_perturbations:
                break
                
        return list(block_set)

    def two_block_swap_search(blocks, seed):
        """
        Try to replace two blocks (bi, bj) with a single block bk that covers
        all unique t-subsets of bi and bj. This is a 2-for-1 improvement.
        """
        rng = random.Random(seed)
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        
        for bi in block_set:
            for ts_idx in block_tsubs[bi]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            n_checked = 0
            for i, bi in enumerate(current_blocks):
                if time.time() > deadline - 3:
                    break
                if bi not in block_set:
                    continue
                    
                if len(block_set) < 2:
                    break
                
                # Pick a random partner
                others = [b for b in block_set if b != bi]
                if not others:
                    continue
                bj = rng.choice(others)
                
                n_checked += 1
                if n_checked > 15000: # Increased limit due to faster checks
                    break
                
                # Compute required t-subsets
                # Unique to bi, unique to bj, and shared with count 2
                req_bi = [ts for ts in block_tsubs[bi] if cover_count[ts] == 1]
                req_bj = [ts for ts in block_tsubs[bj] if cover_count[ts] == 1]
                
                required = set(req_bi)
                required.update(req_bj)
                
                # Check intersection for count==2
                # If a t-subset is in both bi and bj, and cover_count is 2, it becomes unique to the pair
                inter = block_tsub_sets[bi] & block_tsub_sets[bj]
                for ts in inter:
                    if cover_count[ts] == 2:
                        required.add(ts)
                
                if not required:
                    continue
                
                # Find candidate blocks that cover all ts in required
                # Use rarest strategy
                min_ts = min(required, key=lambda x: tsub_block_lens[x])
                
                found_swap = False
                for bk in tsub_blocks[min_ts]:
                    if bk in block_set or bk == bi or bk == bj:
                        continue
                    
                    # Check if bk covers all required tsubs
                    if all(ts in block_tsub_sets[bk] for ts in required):
                        # Perform swap: remove bi, bj; add bk
                        block_set.remove(bi)
                        for ts in block_tsubs[bi]:
                            cover_count[ts] -= 1
                            
                        block_set.remove(bj)
                        for ts in block_tsubs[bj]:
                            cover_count[ts] -= 1
                            
                        block_set.add(bk)
                        for ts in block_tsubs[bk]:
                            cover_count[ts] += 1
                            
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break # Restart while loop
                    
        return list(block_set)

    def final_polish(blocks, seed):
        """
        Aggressive final polishing: repeatedly try all 1-for-1 and 2-for-1 swaps
        until no improvement is found.
        """
        # This is essentially local_search + two_block_swap_search run intensively
        # We just call local_search with a high perturbation count and then two_block_swap
        if time.time() > deadline - 2:
            return blocks
            
        blocks = local_search(blocks, seed)
        
        if time.time() < deadline - 2:
            blocks = two_block_swap_search(blocks, seed + 1)
            
        # One more pass of local search to clean up
        if time.time() < deadline - 1:
            blocks = local_search(blocks, seed + 2)
            
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_trials = 30 # Reduced to allow more time per trial for better local search
    
    for i in range(num_trials):
        if time.time() > deadline - 20:
            break
            
        blocks = greedy_cover(i)
        if not blocks:
            continue
            
        # Local search (single swaps + perturbations)
        blocks = local_search(blocks, i)
        
        # Two-block swap search
        if time.time() < deadline - 10:
            blocks = two_block_swap_search(blocks, i + 1000)
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Final polish on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = final_polish(best_blocks, 77777)
        
    # Fallback if something went wrong
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        
    return [all_blocks[i] for i in best_blocks]
