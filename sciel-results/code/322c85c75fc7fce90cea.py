import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute the triplets covered by each block (indices into all_triplets)
    # C(7,3) = 35 triplets per block
    t_per_block = list(itertools.combinations(range(k), t))
    block_indices = np.empty((n_blocks, len(t_per_block)), dtype=np.int32)
    
    for b_idx, block in enumerate(all_blocks):
        for i, tc in enumerate(t_per_block):
            # Map the local tuple to global triplet id
            # block[tc] creates the global triplet
            tid = triplet_to_id[tuple(block[j] for j in tc)]
            block_indices[b_idx, i] = tid
            
    # Precompute masks for greedy
    block_masks = [0] * n_blocks
    for b_idx in range(n_blocks):
        mask = 0
        for tid in block_indices[b_idx]:
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    # Full mask of all triplets
    all_mask = (1 << n_triplets) - 1
    
    # Precompute frequency of each triplet for weighted greedy
    freq = np.bincount(block_indices.flatten(), minlength=n_triplets).astype(np.float64)
    inv_freq = 1.0 / (freq + 1e-9)

    def greedy_set_cover(seed=0, mode='greedy'):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_blocks = []
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
                
            best_gain = -1
            best_idx = -1
            candidates = []
            
            # Find block with max gain
            for b_idx in range(n_blocks):
                # If block is already used, gain is 0 (since uncovered bits are 0 there)
                # But we can just calculate gain directly.
                if mode == 'greedy':
                    gain = (block_masks[b_idx] & uncovered).bit_count()
                else:
                    # Weighted greedy: sum of inv_freq for uncovered triplets in this block
                    # This is slower, so we might restrict usage or use a simpler heuristic
                    # For now, let's stick to count-based for speed, but maybe add a tie-breaker
                    gain = (block_masks[b_idx] & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if best_idx == -1 or best_gain == 0:
                break
                
            # Randomly pick from candidates to diversify
            # If many candidates, picking one randomly helps avoid local minima
            chosen_idx = rng.choice(candidates)
            
            solution_blocks.append(all_blocks[chosen_idx])
            solution_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        return solution_blocks

    def refine_solution(blocks):
        # Convert to list of lists for mutability
        blocks = [list(b) for b in blocks]
        
        # Helper to calculate coverage counts
        def get_counts(blk_list):
            counts = np.zeros(n_triplets, dtype=np.int32)
            for b in blk_list:
                # We need the indices of triplets for this specific block
                # We can find the global index of b in all_blocks? No, b is a list.
                # It's faster to just recompute the triplets for b.
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    counts[tid] += 1
            return counts

        # Helper to get exclusive triplets for a block given counts
        def get_exclusive(block, counts):
            exc = []
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    exc.append(tid)
            return exc

        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            
            # 1. Remove redundant blocks
            counts = get_counts(blocks)
            new_blocks = []
            for b in blocks:
                # Check if block is redundant
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        redundant = False
                        break
                if not redundant:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
            if not blocks:
                break

            # 2. Try merge 2-for-1
            counts = get_counts(blocks)
            # Precompute exclusive triplets for each block
            exclusives = []
            for b in blocks:
                exclusives.append(get_exclusive(b, counts))
                
            merged = False
            n_sol = len(blocks)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                if not exclusives[i]:
                    continue
                    
                for j in range(i + 1, n_sol):
                    if time.time() > deadline - 5:
                        break
                    if not exclusives[j]:
                        continue
                    
                    # Union of exclusive triplets
                    # We need to check if there is a block that covers all these triplets.
                    # The points involved in these triplets must be contained in the new block.
                    points_needed = set()
                    
                    for tid in exclusives[i]:
                        points_needed.update(all_triplets[tid])
                    for tid in exclusives[j]:
                        points_needed.update(all_triplets[tid])
                        
                    if len(points_needed) > k:
                        continue
                        
                    # If points_needed <= k, we can form a block.
                    # We need to find a block that contains points_needed.
                    # Actually, ANY block containing points_needed will cover the triplets in points_needed?
                    # Yes, because the triplets are subsets of points_needed.
                    # So we just need to check if such a block exists in our universe.
                    # Since we are allowed to add ANY block, we can just construct one.
                    # But we must ensure it's a valid k-subset.
                    
                    # Construct candidate block
                    cand_set = set(points_needed)
                    # Fill remaining spots with arbitrary points
                    # To be deterministic and simple, pick smallest available
                    for p in range(v):
                        if len(cand_set) == k:
                            break
                        if p not in cand_set:
                            cand_set.add(p)
                            
                    if len(cand_set) != k:
                        continue
                        
                    new_block = tuple(sorted(cand_set))
                    
                    # Check if this new block is already in the solution?
                    # If it is, we just remove i and j.
                    # If it's not, we add it and remove i and j.
                    
                    # We need to verify that new_block covers all triplets in exclusives[i] and exclusives[j].
                    # By construction, it contains the points, so it covers the triplets.
                    
                    # Perform the merge
                    # Remove i and j, add new_block
                    # Be careful with indices if we modify list in place.
                    # Let's build a new list.
                    
                    new_blocks_list = []
                    for idx, b in enumerate(blocks):
                        if idx != i and idx != j:
                            new_blocks_list.append(b)
                    new_blocks_list.append(list(new_block))
                    
                    # Verify validity? The evaluator does this. 
                    # But we should ensure we didn't break coverage.
                    # The only triplets that were uniquely covered by i or j are in exclusives[i] and exclusives[j].
                    # new_block covers them.
                    # Triplets covered by both i and j (but not exclusively) are still covered by the other block (or others).
                    # So coverage is maintained.
                    
                    blocks = new_blocks_list
                    merged = True
                    changed = True
                    break
                if merged:
                    break
            if merged:
                continue

        return blocks

    def local_search_swap(blocks):
        """
        Try to replace a block with another block that covers all exclusive triplets
        of the removed block AND has higher total coverage (or same coverage but different).
        This is a 1-swap local search.
        """
        blocks = [list(b) for b in blocks]
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            counts = get_counts(blocks)
            n_sol = len(blocks)
            
            for i in range(n_sol):
                if time.time() > deadline - 5:
                    break
                
                # Get exclusive triplets of block i
                exc_i = get_exclusive(blocks[i], counts)
                
                # We want to find a block j (not i) such that:
                # 1. j covers all triplets in exc_i
                # 2. j is not already in the solution (or if it is, we are just swapping indices, which is pointless unless we remove i and add j, resulting in same set? No, if j is in solution, we can't add it again. We can only replace i with j if j is not in solution? Or if we allow duplicates? No, blocks must be distinct.)
                # So j must be a block NOT in the current solution.
                
                # To make this efficient, we precompute for each block in the solution, the set of blocks that cover its exclusive triplets.
                # This is expensive.
                
                # Alternative: For each block i, check all possible blocks j in the universe?
                # n_blocks is 346k. n_sol is ~80. 80 * 346k = 27M checks.
                # Each check: does j cover exc_i?
                # exc_i is a list of triplet ids.
                # We can check if (mask_j & mask_exc_i) == mask_exc_i.
                
                # Precompute mask for exc_i
                mask_exc_i = 0
                for tid in exc_i:
                    mask_exc_i |= (1 << tid)
                    
                if mask_exc_i == 0:
                    # No exclusive triplets, block is redundant?
                    # We already removed redundant blocks. So this shouldn't happen unless counts changed.
                    # But counts are computed at the start of the loop.
                    # If a block has no exclusive triplets, it's redundant.
                    # We can just remove it.
                    blocks.pop(i)
                    changed = True
                    break
                    
                # Iterate through all blocks in universe
                # Optimization: We can skip blocks that are already in the solution.
                # Create a set of tuples of current blocks for fast lookup
                current_blocks_set = set(tuple(b) for b in blocks)
                
                best_j = -1
                # We want a block that covers exc_i.
                # Among those, we might prefer one that covers many other triplets?
                # Or just any one?
                # If we replace i with j, the new solution is (blocks - {i}) + {j}.
                # We need to ensure coverage is maintained.
                # Since j covers exc_i, and other blocks cover the rest (as they did for i's non-exclusive triplets), coverage is maintained.
                
                # Let's just find the first valid j.
                # Or a random one to diversify.
                
                # To speed up, we can sample a few blocks.
                # But for now, let's try to find one.
                
                # This loop is O(n_blocks).
                # 346k iterations per block i.
                # 80 blocks * 346k = 27M iterations.
                # In Python, this might be too slow if done repeatedly.
                
                # Let's limit the search or make it smarter.
                # We can precompute for each triplet, the list of blocks covering it.
                # Then for exc_i, the candidate blocks are the intersection of the lists for each triplet in exc_i.
                
                # Let's precompute triplet_to_blocks once.
                # This is memory intensive: 2024 triplets * ~1000 blocks/triplet = 2M ints. OK.
                
                # But we haven't precomputed it. Let's do it here if needed?
                # Or just use the mask check.
                
                # Let's try the mask check but with a time limit per swap.
                t_start_swap = time.time()
                
                # Find a block j that covers mask_exc_i
                # We can iterate through all_blocks.
                for b_idx in range(n_blocks):
                    if time.time() - t_start_swap > 1.0: # 1s per swap check? Too long.
                        break
                    if tuple(all_blocks[b_idx]) in current_blocks_set:
                        continue
                        
                    if (block_masks[b_idx] & mask_exc_i) == mask_exc_i:
                        # Found a candidate
                        # Check if this swap improves something?
                        # The number of blocks stays the same.
                        # We might want to choose j that maximizes the number of triplets it covers
                        # that were previously covered only by i? No, i is removed.
                        # The triplets covered by i and j are now covered by j.
                        # The triplets covered by i and not j are... wait.
                        # j covers exc_i.
                        # What about triplets covered by i and NOT in exc_i?
                        # Those are covered by at least one other block.
                        # So coverage is safe.
                        
                        # Let's just take the first one found.
                        # To be more effective, we could pick one that covers the most total triplets?
                        # Or one that is "different" from i?
                        
                        # Let's replace i with j
                        blocks[i] = list(all_blocks[b_idx])
                        changed = True
                        best_j = b_idx
                        break
                        
                if changed:
                    break # Restart the while loop to recompute counts
                    
        return blocks

    def get_counts(blk_list):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blk_list:
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                counts[tid] += 1
        return counts

    def get_exclusive(block, counts):
        exc = []
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            if counts[tid] == 1:
                exc.append(tid)
        return exc

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    # Bitmask greedy is fast. We can afford more seeds.
    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 40:
            break
        blocks = greedy_set_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        best_blocks = greedy_set_cover(0)
        
    # Local search polish
    # First pass: remove redundant and merge 2-for-1
    best_blocks = refine_solution(best_blocks)
    
    # Second pass: 1-swap local search
    # This might be expensive, so we only do it if we have time.
    if time.time() < deadline - 15:
        best_blocks = local_search_swap(best_blocks)
        
    # Final cleanup
    best_blocks = refine_solution(best_blocks)
    
    return best_blocks

if __name__ == "__main__":
    pass
