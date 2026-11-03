import itertools
import time
import random
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Bitset representation
    # tsub_mask[b] is a Python int where bit i is 1 if tsub i is covered by block b
    tsub_mask = [0] * n_blocks
    # tsub_blocks_mask[i] is a Python int where bit b is 1 if block b covers tsub i
    # n_blocks is ~50k, so this int is ~50k bits. Python handles this.
    tsub_blocks_mask = [0] * n_tsubsets
    
    # List of tsub indices for each block (for cover_count updates)
    block_tsubs = [[] for _ in range(n_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask
        
    full_tsub_mask = (1 << n_tsubsets) - 1
    
    def greedy_cover(seed):
        """
        Heap-based lazy greedy.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []
        
        # Initialize heap with (gain, block_idx)
        # We need to store negative gain for max-heap behavior, or just use min-heap and negate
        # Actually, heapq is min-heap. We want max gain. So push (-gain, b_idx).
        # But gains change.
        # Lazy approach:
        # 1. Push all blocks with initial gain.
        # 2. Pop min (-g, b).
        # 3. Check if -g is still the correct gain for b given current uncovered_mask.
        # 4. If not, recompute and push new value.
        # 5. If yes, select b.
        
        heap = []
        for b_idx in range(n_blocks):
            gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
            heapq.heappush(heap, (-gain, b_idx))
            
        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
                
            # Pop until we find a valid block
            while heap:
                neg_gain, b_idx = heapq.heappop(heap)
                current_gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                
                # If the gain in the heap is stale (higher than current), re-push
                if -neg_gain > current_gain:
                    heapq.heappush(heap, (-current_gain, b_idx))
                    continue
                    
                # If current gain is 0, no block can cover more, but we know uncovered_mask is not 0?
                # If all gains are 0 but uncovered_mask is not 0, it means no block covers any uncovered subset?
                # This shouldn't happen if the problem is solvable, but if it does, we break.
                if current_gain == 0:
                    # No block covers any remaining uncovered subset?
                    # This implies the greedy got stuck or the problem instance is tricky.
                    # In a covering design, every t-subset is covered by some block.
                    # If uncovered_mask has bits, and all blocks have 0 gain, it means those bits are not covered by ANY block?
                    # That's impossible if all_blocks is complete.
                    # Wait, if we selected blocks, we removed their coverage.
                    # If current_gain is 0 for the best block, it means NO block covers any uncovered subset.
                    # This means the algorithm failed to find a cover? Or we exhausted blocks?
                    # Actually, if we just pop a block with gain 0, it means it covers nothing new.
                    # We should skip it? No, if the best block has gain 0, we are stuck.
                    # But in a valid covering design problem, a cover exists.
                    # If we are stuck, it means our greedy choice led to a dead end?
                    # Greedy set cover doesn't get stuck in terms of "no block covers anything" unless the universe is not fully coverable by the available sets.
                    # Here, the available sets are all k-subsets. They cover everything.
                    # So gain 0 means we are done? No, `uncovered_mask` is not 0.
                    # This case should be rare or impossible if we start with full coverage.
                    # Let's just break to be safe.
                    break
                
                # Valid block found
                selected.append(b_idx)
                uncovered_mask &= ~tsub_mask[b_idx]
                
                # We don't need to update the heap immediately.
                # The lazy check will handle stale entries.
                # However, we must ensure we don't pick the same block again.
                # We can mark selected blocks or just let the gain become 0 for them?
                # If we select b, its gain becomes 0 relative to new uncovered_mask.
                # If it's in the heap, it will be popped eventually and re-pushed with 0.
                # To avoid re-processing, we can just let it be.
                # But we might pop it again and push it again.
                # Optimization: If we select b, we can ignore it in future pops.
                # We can't easily remove from heap.
                # But if we pop b again, current_gain will be 0 (since its subsets are covered).
                # So it will be pushed back with 0.
                # If we pop many 0-gain blocks, we might waste time.
                # We can stop if the best available gain is 0.
                
                # To prevent infinite loops or excessive 0-gain pops:
                # If we popped a block and its gain was > 0, we selected it.
                # The next pop might be a stale entry for a block we already selected?
                # No, we didn't remove it.
                # If we pop a block we already selected, its current_gain is 0.
                # The heap entry might have a high negative gain (stale).
                # We re-push with 0.
                # Eventually, all selected blocks will be in the heap with gain 0.
                # The top of the heap will be the max gain among *unselected* blocks (or stale unselected).
                
                # We break the inner while loop because we found a valid block to select.
                break
            
            if not heap or (heap and -heap[0][0] == 0):
                # If heap is empty or best gain is 0, we are stuck.
                break

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        Uses bitsets to find candidate replacement blocks quickly.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        
        # Initialize cover counts
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)
                    
            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue # Restart loop as structure changed
            
            # Phase 2: Try swaps
            # We try to swap a block b with a block r that covers all unique subsets of b.
            # This might not reduce size, but it might allow future deletions.
            # Or, we look for 2-for-1 swaps.
            
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue
                    
                # Find unique subsets for b
                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)
                        
                if not unique_tsubs:
                    # Should have been caught in Phase 1, but safe check
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break # Restart
                    
                # Find candidate blocks covering all unique_tsubs
                # candidate_mask = AND of tsub_blocks_mask[ts] for ts in unique_tsubs
                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                
                if cand_mask == 0:
                    continue
                    
                # Remove current block b from candidates
                cand_mask &= ~(1 << b)
                
                # Extract set bits from cand_mask
                candidates = []
                m = cand_mask
                while m:
                    # Get lowest set bit
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 20: # Limit candidates
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                
                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue
                        
                    # Perform swap b -> r
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                        
                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1
                        
                    improved = True
                    swapped = True
                    break # Restart local search loop to check for new deletions
                    
                if swapped:
                    break # Restart while loop
                    
        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks
            
        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)
            
            # Check pairs
            # To limit O(N^2), we can check random pairs or pairs involving blocks with many unique subsets.
            # Let's try random pairs.
            n_checks = 100
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break
                    
                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                if b1 not in block_set or b2 not in block_set:
                    continue
                    
                # Find unique subsets for b1 and b2
                # Note: A subset might be unique to b1, or unique to b2, or covered by both but no one else.
                # For 2-for-1, the replacement block r must cover all subsets that are ONLY covered by b1 or ONLY covered by b2.
                # Subsets covered by both b1 and b2 (and no one else) are also "unique to the pair".
                
                # Let U1 be subsets with count 1 in b1 (relative to current cover_count)
                # Let U2 be subsets with count 1 in b2
                # Actually, if a subset is in b1 and b2, and count is 2, removing both makes count 0.
                # So the set of subsets that MUST be covered by r is:
                # { s | s in b1 and count[s] <= 1 } U { s | s in b2 and count[s] <= 1 }
                # Wait, if s is in b1 and b2, count[s] is at least 2.
                # If we remove b1 and b2, count[s] drops by 2.
                # If count[s] was 2, it becomes 0. So r MUST cover s.
                # If count[s] was 3, it becomes 1. So r does NOT strictly need to cover s, but if it does, it's fine.
                # To be safe and simple, let's require r to cover all s such that (s in b1 and count[s] <= 1) OR (s in b2 and count[s] <= 1).
                # Actually, if s in b1 and b2, and count[s]=2, then s is "unique to the pair".
                
                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                        
                if required_mask == 0:
                    # Both blocks are redundant? Should have been caught.
                    continue
                    
                # Find blocks covering required_mask
                # cand_mask = AND of tsub_blocks_mask[ts] for all ts in required_mask
                # This is expensive if required_mask has many bits.
                # We can iterate over the set bits of required_mask.
                
                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb
                    
                if cand_mask == 0:
                    continue
                    
                # Exclude b1 and b2
                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)
                
                # Get candidates
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break
                        
                if not candidates:
                    continue
                    
                rng.shuffle(candidates)
                r = candidates[0]
                
                # Perform swap: remove b1, b2; add r
                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1
                    
                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1
                    
                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1
                    
                improved = True
                break # Restart while loop to check for new deletions
                
        return list(block_set)

    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    # 1. Greedy with different seeds
    # 2. Local search on the result
    # 3. 2-for-1 search on the result
    
    num_restarts = 20 # Increased from 10 due to faster greedy
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Greedy
        seed = 1000 + i
        greedy_sol = greedy_cover(seed)
        
        if not greedy_sol:
            continue
            
        # Local Search
        ls_sol = local_search(greedy_sol, seed)
        
        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)
        
        # Final deletion pass
        # (local_search and two_for_one_search already do deletion, but just in case)
        
        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()
            
    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)
        
    return [list(all_blocks[idx]) for idx in best_solution]
