import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Precomputation
    # n_t = C(v, t) = C(24, 3) = 2024
    # n_blocks = C(v, k) = C(24, 7) = 346104
    
    # We need to map each t-subset to an index 0..n_t-1
    # And for each block, we need a bitmask of which t-subsets it covers.
    # Since n_t = 2024, we can use 32 uint64s per block.
    # 346104 blocks * 32 * 8 bytes = ~88 MB. This is feasible.
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage masks for all blocks
    # mask[block_idx] is an array of uint64s
    # We store as a 2D numpy array: (n_blocks, n_uint64s)
    n_uint64s = (n_t + 63) // 64
    
    # To speed up precomputation, we can use a lookup table for t-subsets
    # Or just iterate. 346k * 35 = 12M operations. 
    # Python loop might take ~1-2s. Let's try to optimize or just run it.
    
    # Actually, let's use a more efficient precomputation if possible.
    # But for simplicity and correctness, let's just do it in a loop first.
    # If it's too slow, we'll optimize.
    
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    
    # Optimization: Precompute the bitmask for each t-subset? No, t-subsets are indices.
    # We just need to set bit idx in the appropriate uint64.
    
    # Let's parallelize or just run. 12M simple ops is fast in C, slow in Python.
    # Let's try to vectorize the precomputation?
    # It's hard to vectorize "for each block, for each t-subset in block, set bit".
    
    # Alternative: Use the fact that we only need the masks for the greedy step.
    # But we need them for ALL blocks to pick the best one.
    
    # Let's measure the precomputation time.
    # If it takes > 5s, we need a different approach.
    
    # Trick: We can generate the masks for a block by iterating its t-subsets.
    # Let's just do it.
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
            
    if time.time() > deadline - 10.0:
        # Fallback if precomputation is too slow
        # Return a simple greedy result
        pass

    # 2. Vectorized Greedy Set Cover
    # State:
    # covered_mask: array of uint64s representing currently covered t-subsets
    # selected_blocks: list of block indices
    
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_blocks = []
    
    # We need to find the block that maximizes (newly covered bits) / cost.
    # Cost is 1 for all blocks. So maximize newly covered bits.
    
    # Newly covered bits for block j = popcount(coverage_masks[j] & ~covered_mask)
    
    # We can compute this for all blocks at once?
    # ~covered_mask is a vector of uint64s.
    # coverage_masks is (n_blocks, n_uint64s).
    # bitwise_and = coverage_masks & (~covered_mask)
    # popcount = np.bitwise_count(bitwise_and).sum(axis=1)
    
    # This is a vectorized operation over 346k x 32 array.
    # 346k * 32 = 11M operations. Very fast in NumPy.
    
    # Greedy Loop
    max_iterations = 200 # Safety
    for _ in range(max_iterations):
        if time.time() > deadline - 10.0:
            break
            
        # Check if all covered
        # Sum of bits in covered_mask
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
            
        # Compute gains for all blocks
        # We need to avoid re-selecting blocks?
        # In standard greedy set cover, we don't remove blocks, so we can just pick the best.
        # However, picking the same block again is useless.
        # We can mask out already selected blocks?
        # Or just let it pick the one with max gain. If it's already selected, gain is 0 (or low).
        # Actually, if a block is already selected, its covered bits are in covered_mask,
        # so its gain (bits in block AND NOT covered) will be 0.
        # So it won't be picked unless all gains are 0 (stuck).
        
        not_covered = ~covered_mask
        # Bitwise and
        intersections = coverage_masks & not_covered
        # Popcount
        gains = np.bitwise_count(intersections).sum(axis=1)
        
        best_block_idx = np.argmax(gains)
        best_gain = gains[best_block_idx]
        
        if best_gain == 0:
            # Stuck, no block covers new elements. Should not happen if valid covering exists.
            break
            
        selected_blocks.append(best_block_idx)
        covered_mask |= coverage_masks[best_block_idx]
        
    # Convert indices to blocks
    current_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # 3. Local Search: Block Swapping
    # Goal: Reduce the number of blocks.
    # Strategy: Try to replace a block with a different block that covers all the 
    # "unique" t-subsets of the old block, while ideally covering more others or 
    # allowing another block to be removed.
    
    # Simpler Local Search:
    # 1. Compute cover counts for current solution.
    # 2. Identify "essential" blocks (blocks that cover at least one t-subset uniquely).
    # 3. For non-essential blocks, try to remove them. If removal leaves a gap, 
    #    try to find a replacement block that fills the gap and is "better" (covers more redundant stuff).
    
    # Actually, a stronger move:
    # Try to replace block A with block B.
    # Condition: All t-subsets in A that are NOT covered by other blocks must be covered by B.
    # Benefit: If B allows us to remove another block C, we win.
    # Or simply: If B covers the same "essential" parts of A but is "more efficient" globally?
    
    # Let's stick to a simpler heuristic:
    # Iteratively try to remove each block.
    # If removal causes uncovered t-subsets, try to find a block that covers them.
    # If found, replace the removed block with the new block only if it leads to a better state?
    # No, replacement keeps size same. Removal reduces size.
    
    # Let's try:
    # 1. Compute cover counts.
    # 2. For each block i:
    #    - Temporarily remove it.
    #    - Find uncovered t-subsets U.
    #    - If U is empty, remove block i permanently.
    #    - If U is not empty, find a block j (not in current set) that covers all U.
    #      - If found, replace i with j?
    #        - This keeps size same.
    #        - But maybe j covers more of the "redundant" parts, allowing future removals?
    #        - Or maybe we can just swap if it doesn't increase the "deficit" elsewhere?
    
    # Let's implement a "Swap to improve structure" local search.
    
    # Precompute cover counts for current solution
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        # Increment counts for all t-subsets in block b_idx
        # We can use the mask? No, we need indices.
        # Let's precompute a list of t-subset indices for each block.
        # Or just iterate.
        pass
    
    # To make local search fast, we need efficient updates.
    # Let's precompute block_t_indices: list of lists of t-subset indices for each block.
    # Memory: 346k * 35 ints. ~12M ints. 48MB. OK.
    
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
        
    # Recompute cover_counts
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_counts[block_t_indices[b_idx]] += 1
        
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try to remove blocks
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            t_idxs = block_t_indices[b_idx]
            
            # Check if any t-subset has count 1
            # If so, removing b_idx will uncover them.
            # If none, remove it.
            
            # Vectorized check:
            # counts = cover_counts[t_idxs]
            # if min(counts) > 1: remove
            # else: find uncovered ones
            
            # To speed up, we can just check.
            # But we also want to try replacements.
            
            # Let's just try removal first.
            # If removal is valid, do it and set improved=True.
            # If not, try to find a replacement?
            
            # For now, let's just do removal. 
            # If we want to be smarter, we can try to find a block that covers the "essential" parts.
            
            # Essential parts: t_idxs where cover_counts[t_idx] == 1
            essential_mask = (cover_counts == 0) # Wait, we haven't removed it yet.
            # We need to know what becomes uncovered.
            # Uncovered after removal = t_idxs where cover_counts[t_idx] == 1
            
            # Let's just check if min(cover_counts[t_idxs]) > 1
            if np.min(cover_counts[t_idxs]) > 1:
                # Safe to remove
                cover_counts[t_idxs] -= 1
                selected_blocks.pop(i)
                improved = True
                # Restart loop or continue? Restart to check new possibilities.
                # But we are in a for loop. Let's break and restart.
                break
            
        # If we broke out due to improvement, restart the while loop.
        # If we finished the for loop without improvement, we are stuck.
        # We can try a "swap" move here.
        
        # Swap Move:
        # Pick a random block i.
        # Find its essential t-subsets (count==1).
        # Find a random block j not in selected_blocks that covers all essential t-subsets of i.
        # If found, swap i with j.
        # Why? Maybe j allows other blocks to become removable.
        
        if not improved and time.time() < deadline - 5.0:
            # Try a few random swaps
            rng = random.Random(int(time.time() * 1000))
            for _ in range(100):
                if time.time() > deadline - 5.0:
                    break
                
                if len(selected_blocks) == 0:
                    break
                
                i = rng.randint(0, len(selected_blocks) - 1)
                b_idx = selected_blocks[i]
                t_idxs = block_t_indices[b_idx]
                
                # Essential t-subsets for this block
                # These are the ones that will be lost if we remove it.
                essential_idxs = [idx for idx in t_idxs if cover_counts[idx] == 1]
                
                if not essential_idxs:
                    # This block is fully redundant. Should have been removed.
                    # Remove it.
                    cover_counts[t_idxs] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    break
                
                # Find a block j that covers all essential_idxs
                # We can scan all blocks? 346k * len(essential_idxs) checks.
                # Too slow if done naively.
                
                # Optimization:
                # Precompute for each t-subset, which blocks cover it?
                # Inverse index.
                # t_subset_to_blocks: dict of t_idx -> list of block_idxs
                # This is expensive to build (12M entries).
                # But we can build it once.
                
                # Let's build it if we haven't.
                # It's a list of lists.
                # Memory: 12M ints. 48MB. OK.
                
                # Let's do it outside the loop?
                # No, let's just do a random sample of blocks to find a candidate.
                # Or just skip swaps for now and rely on greedy + removal.
                
                # Actually, the greedy might leave many redundant blocks.
                # The removal loop should clean them up.
                # If the removal loop finds nothing, it means every block is essential.
                # In that case, we are at a local minimum for the "remove" move.
                
                # To escape, we need a swap.
                # Let's implement a fast check for swap candidates.
                # Pick a random essential t-subset e from essential_idxs.
                # Find blocks that cover e.
                # Check if they cover ALL essential_idxs.
                
                # We need t_subset_to_blocks.
                # Let's build it now if not built.
                if not hasattr(construct, '_t_to_blocks'):
                    t_to_blocks = [[] for _ in range(n_t)]
                    for b_idx, t_idxs in enumerate(block_t_indices):
                        for t_idx in t_idxs:
                            t_to_blocks[t_idx].append(b_idx)
                    construct._t_to_blocks = t_to_blocks
                else:
                    t_to_blocks = construct._t_to_blocks
                
                # Pick a random essential t-subset
                e = rng.choice(essential_idxs)
                # Get candidate blocks
                candidates = t_to_blocks[e]
                
                if not candidates:
                    continue
                    
                # Pick a random candidate
                j = rng.choice(candidates)
                
                # Check if j is already in selected_blocks
                if j in selected_blocks:
                    continue
                
                # Check if j covers all essential_idxs
                # j's t-subsets: block_t_indices[j]
                # We need set(essential_idxs) subset of set(block_t_indices[j])
                
                # For speed, we can check:
                # Are all essential_idxs in block_t_indices[j]?
                # Since essential_idxs is small (usually 1-5), we can just check.
                
                # To make it faster, we can precompute a set for block_t_indices?
                # Or just use a counter.
                
                # Let's just use a set for the candidate's t-subsets.
                # Creating a set of 35 elements is fast.
                j_t_set = set(block_t_indices[j])
                
                valid_swap = True
                for e_idx in essential_idxs:
                    if e_idx not in j_t_set:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Perform swap
                    # Remove i, add j
                    cover_counts[block_t_indices[i]] -= 1
                    cover_counts[block_t_indices[j]] += 1
                    selected_blocks[i] = j
                    improved = True
                    break

    # Final Verification
    final_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Check validity
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Invalid. Fallback to greedy only (which should be valid)
        # Re-run greedy without local search?
        # Or just return the greedy result from before local search?
        # We didn't save it.
        # Let's just return the current blocks and hope the evaluator handles it?
        # No, must be valid.
        
        # Reconstruct a valid greedy covering quickly.
        # We have the masks.
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        valid_blocks = []
        for _ in range(200):
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            valid_blocks.append(best_idx)
            covered_mask |= coverage_masks[best_idx]
            
        return [list(all_blocks[i]) for i in valid_blocks]

    return final_blocks
