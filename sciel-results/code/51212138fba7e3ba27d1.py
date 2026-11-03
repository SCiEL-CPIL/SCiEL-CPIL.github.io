import itertools
import random
import time
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback for non-target parameters
    if v != 19 or k != 7 or t != 4:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            best_count = -1
            # This is very slow for large v, but only used if not 19,7,4
            for block in itertools.combinations(range(v), k):
                count = 0
                for tsub in itertools.combinations(block, t):
                    if tsub in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = block
            if best_block is None: break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                # Try to finish quickly or return invalid? Must return valid.
                # This fallback is risky. But task is 19,7,4.
                pass
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    # 1. Precomputation
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks = []
    block_tsubs = [] # list of lists of tsub ids
    tsub_to_blocks = [[] for _ in range(num_tsubs)] # inverse index
    
    for block_tuple in itertools.combinations(points, k):
        block_id = len(all_blocks)
        all_blocks.append(block_tuple)
        tsubs_in_block = []
        for tsub in itertools.combinations(block_tuple, t):
            tid = tsub_to_id[tsub]
            tsubs_in_block.append(tid)
            tsub_to_blocks[tid].append(block_id)
        block_tsubs.append(tsubs_in_block)
        
    num_blocks_total = len(all_blocks)
    
    # 2. Greedy Initialization
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    current_blocks = []
    in_cover = np.zeros(num_blocks_total, dtype=bool)
    
    def is_redundant(b_id):
        for tid in block_tsubs[b_id]:
            if cover_count[tid] <= 1:
                return False
        return True

    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
        
        # Find block with max gain
        best_block = -1
        best_gain = -1
        
        for b_id in range(num_blocks_total):
            if in_cover[b_id]:
                continue
            gain = 0
            for tid in block_tsubs[b_id]:
                if cover_count[tid] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_id
                
        if best_block == -1 or best_gain == 0:
            break
            
        current_blocks.append(best_block)
        in_cover[best_block] = True
        for tid in block_tsubs[best_block]:
            cover_count[tid] += 1
            
        # Check if all covered
        if np.min(cover_count) > 0:
            break
            
    # 3. Local Search: Redundancy Removal + Swaps + 2-Swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Remove Redundant Blocks
        for i in range(len(current_blocks) - 1, -1, -1):
            b_id = current_blocks[i]
            if is_redundant(b_id):
                for tid in block_tsubs[b_id]:
                    cover_count[tid] -= 1
                in_cover[b_id] = False
                current_blocks.pop(i)
                improved = True
                
        # Pass 2: Try 1-Swaps
        swaps_made = 0
        max_swaps = 20
        rng = random.Random(42)
        
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for idx in indices:
            if swaps_made >= max_swaps:
                break
            if time.time() > deadline:
                break
                
            b_id = current_blocks[idx]
            
            # Tentative removal
            for tid in block_tsubs[b_id]:
                cover_count[tid] -= 1
            
            # Find uncovered
            uncovered_tids = [tid for tid in range(num_tsubs) if cover_count[tid] == 0]
            
            if not uncovered_tids:
                in_cover[b_id] = False
                current_blocks.pop(idx)
                improved = True
                continue
            
            # Find candidate b' that covers all uncovered_tids
            candidates = None
            for tid in uncovered_tids:
                blocks_covering_tid = tsub_to_blocks[tid]
                if candidates is None:
                    candidates = set(blocks_covering_tid)
                else:
                    candidates.intersection_update(blocks_covering_tid)
                    
            if candidates:
                valid_candidates = [c for c in candidates if not in_cover[c]]
                if valid_candidates:
                    b_prime = rng.choice(valid_candidates)
                    
                    for tid in block_tsubs[b_prime]:
                        cover_count[tid] += 1
                    in_cover[b_prime] = True
                    current_blocks[idx] = b_prime
                    in_cover[b_id] = False
                    swaps_made += 1
                else:
                    for tid in block_tsubs[b_id]:
                        cover_count[tid] += 1
            else:
                for tid in block_tsubs[b_id]:
                    cover_count[tid] += 1

        # Pass 3: Try 2-Swaps (New)
        # We try to replace two blocks (i, j) with two new blocks (p, q).
        # This is expensive, so we limit it.
        if time.time() < deadline - 10: # Leave time for final cleanup
            n_blocks = len(current_blocks)
            # Select a subset of blocks to try for 2-swaps to save time
            # We prioritize blocks that are "less essential" (more redundant coverage)
            block_scores = []
            for b_id in current_blocks:
                # Score: number of tsubs in this block that are covered by >1 other block
                redundant_count = 0
                for tid in block_tsubs[b_id]:
                    if cover_count[tid] > 1:
                        redundant_count += 1
                block_scores.append((redundant_count, b_id))
            
            # Sort by redundancy score descending
            block_scores.sort(reverse=True)
            
            # Take top M blocks for 2-swap attempts
            M = min(10, n_blocks)
            candidate_blocks = [b_id for _, b_id in block_scores[:M]]
            
            two_swaps_made = 0
            max_two_swaps = 5
            
            # Iterate over pairs in candidate_blocks
            for i_idx in range(len(candidate_blocks)):
                if two_swaps_made >= max_two_swaps:
                    break
                if time.time() > deadline - 5:
                    break
                    
                b_id1 = candidate_blocks[i_idx]
                
                for j_idx in range(i_idx + 1, len(candidate_blocks)):
                    if two_swaps_made >= max_two_swaps:
                        break
                    if time.time() > deadline - 5:
                        break
                        
                    b_id2 = candidate_blocks[j_idx]
                    
                    # Check if b_id1 and b_id2 are still in current_blocks
                    # (They might have been removed by previous swaps in this pass? 
                    #  No, we didn't modify current_blocks in Pass 2 unless we removed redundant ones, 
                    #  but we re-indexed? No, we modified in place. 
                    #  We should verify they are still there.)
                    if b_id1 not in current_blocks or b_id2 not in current_blocks:
                        continue
                    
                    # Tentatively remove b_id1 and b_id2
                    for tid in block_tsubs[b_id1]:
                        cover_count[tid] -= 1
                    for tid in block_tsubs[b_id2]:
                        cover_count[tid] -= 1
                    
                    # Find uncovered tids
                    uncovered_tids = [tid for tid in range(num_tsubs) if cover_count[tid] == 0]
                    
                    if not uncovered_tids:
                        # Both were redundant? Should have been caught in Pass 1.
                        # If so, remove them.
                        # But we already decremented counts.
                        # We need to remove them from current_blocks.
                        # This is messy. Let's just revert and rely on Pass 1.
                        for tid in block_tsubs[b_id1]:
                            cover_count[tid] += 1
                        for tid in block_tsubs[b_id2]:
                            cover_count[tid] += 1
                        continue
                    
                    # We need to find two blocks p, q such that p U q covers uncovered_tids.
                    # This is a 2-set cover problem on the subset of uncovered_tids.
                    # Heuristic:
                    # 1. Find best single block p that covers most uncovered_tids.
                    # 2. Let remaining = uncovered_tids - covered_by_p.
                    # 3. Find best single block q that covers all remaining.
                    
                    best_p = -1
                    best_p_gain = -1
                    best_p_covered = set()
                    
                    # Limit search for p to blocks that cover at least one uncovered tid
                    # And are not in cover
                    for p in range(num_blocks_total):
                        if in_cover[p]:
                            continue
                        gain = 0
                        covered = set()
                        for tid in block_tsubs[p]:
                            if tid in set(uncovered_tids):
                                gain += 1
                                covered.add(tid)
                        if gain > best_p_gain:
                            best_p_gain = gain
                            best_p = p
                            best_p_covered = covered
                            
                    if best_p == -1 or best_p_gain == 0:
                        # Revert
                        for tid in block_tsubs[b_id1]:
                            cover_count[tid] += 1
                        for tid in block_tsubs[b_id2]:
                            cover_count[tid] += 1
                        continue
                    
                    remaining = set(uncovered_tids) - best_p_covered
                    if not remaining:
                        # p covers all? Then we could have done a 1-swap.
                        # But we are in 2-swap. We can just pick any q?
                        # No, we need to replace 2 with 2.
                        # If p covers all, we can pick any q that is not in cover?
                        # But that increases size if we keep both? No, we replace 2 with 2.
                        # If p covers all, we can pick a random q?
                        # Or better, pick q that maximizes redundancy with others?
                        # For now, let's just find a valid q.
                        pass
                    
                    best_q = -1
                    for q in range(num_blocks_total):
                        if in_cover[q] or q == best_p:
                            continue
                        # q must cover all remaining
                        can_cover = True
                        for tid in remaining:
                            if tid not in block_tsubs[q]:
                                can_cover = False
                                break
                        if can_cover:
                            best_q = q
                            break
                            
                    if best_q != -1:
                        # Perform swap
                        # Add p and q
                        for tid in block_tsubs[best_p]:
                            cover_count[tid] += 1
                        for tid in block_tsubs[best_q]:
                            cover_count[tid] += 1
                        in_cover[best_p] = True
                        in_cover[best_q] = True
                        
                        # Remove b_id1 and b_id2
                        in_cover[b_id1] = False
                        in_cover[b_id2] = False
                        
                        # Update current_blocks
                        # Find indices
                        idx1 = current_blocks.index(b_id1)
                        current_blocks[idx1] = best_p
                        idx2 = current_blocks.index(b_id2)
                        current_blocks[idx2] = best_q
                        
                        two_swaps_made += 1
                        improved = True
                    else:
                        # Revert
                        for tid in block_tsubs[b_id1]:
                            cover_count[tid] += 1
                        for tid in block_tsubs[b_id2]:
                            cover_count[tid] += 1

    # Final Redundancy Removal
    for i in range(len(current_blocks) - 1, -1, -1):
        b_id = current_blocks[i]
        if is_redundant(b_id):
            for tid in block_tsubs[b_id]:
                cover_count[tid] -= 1
            in_cover[b_id] = False
            current_blocks.pop(i)

    final_blocks = [list(all_blocks[b_id]) for b_id in current_blocks]
    
    return final_blocks
