# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses bitset-accelerated greedy with local search (redundant block removal).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets and index them
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, compute its coverage bitmask
    # n_t is 2024 for v=24, t=3. Python ints handle this easily.
    coverage_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        coverage_masks.append(mask)
    
    # Precompute list of t-subsets for each block (for verification/fallback if needed, but mainly for local search logic if we don't use masks there)
    # Actually, we can use masks for local search too.
    
    # For each t-subset, list of block indices that contain it (for potential alternative strategies, but greedy uses masks)
    # This is expensive to build, so we might skip it unless needed for a specific heuristic.
    # For now, we rely on greedy + local search removal.

    def greedy_construct(rng, deadline):
        """
        Randomized greedy: repeatedly pick the block covering the most uncovered t-subsets.
        Uses bitmasks for speed.
        """
        # Initialize uncovered mask to all 1s
        # Create a mask with n_t bits set to 1
        all_ones = (1 << n_t) - 1
        uncovered = all_ones
        blocks = []
        remaining_uncovered = n_t

        while remaining_uncovered > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0
            best_mask = 0
            
            # We need to find the block that maximizes (block_mask & uncovered).bit_count()
            # To speed this up, we can iterate through blocks.
            # Optimization: If we have many blocks, this is still O(n_blocks) per step.
            # n_blocks = 346,104.
            # n_steps ~ 76.
            # 76 * 346,104 ~ 26 million iterations.
            # Each iteration involves a bitwise AND and bit_count.
            # In Python, this might take ~1-2 seconds per restart if optimized?
            # Let's try to optimize the loop.
            
            # Randomize order to break ties and explore different local optima
            # Instead of shuffling the whole list (expensive), we can just iterate and use random tie-breaking
            # Or, we can just iterate all blocks.
            
            # A simple loop is often faster in Python than complex data structures if the operation is simple.
            # Let's try iterating all blocks.
            
            for b_idx in range(n_blocks):
                # Calculate gain
                # gain = (coverage_masks[b_idx] & uncovered).bit_count()
                # If gain > best_gain:
                #    best_gain = gain
                #    best_block = b_idx
                #    best_mask = coverage_masks[b_idx]
                # elif gain == best_gain and gain > 0 and rng.random() < 0.5:
                #    best_block = b_idx
                #    best_mask = coverage_masks[b_idx]
                
                # To avoid calling rng.random() for every tie (which is expensive),
                # we can just take the first one found with max gain, or use a deterministic tie-breaker 
                # like block index. However, randomization helps escape local minima.
                # Let's use a small random subset of blocks or just randomize the start index?
                # Actually, shuffling 346k elements takes ~10ms. 76 steps * 10ms = 0.76s per restart.
                # That's acceptable.
                pass
            
            # Better approach:
            # Iterate through blocks.
            # To add randomness, we can pick a random starting offset or shuffle indices.
            # Let's shuffle indices once per restart? No, per step is better for diversity.
            # But shuffling 346k items every step is too slow.
            
            # Alternative: Just iterate all blocks. The greedy choice is deterministic given the uncovered set.
            # To get different results, we can perturb the greedy choice.
            # Strategy: Find the top few candidates (e.g., top 5) and pick one randomly.
            # Or, just use a fixed seed and rely on the fact that different initial random seeds
            # for the *order of evaluation* if we implement it.
            
            # Let's try a simple optimization:
            # Keep a list of block indices. Shuffle it once at the start of the greedy phase?
            # No, that's not enough.
            
            # Let's just iterate all blocks. It's O(N).
            # We can speed up the inner loop by using local variables.
            masks = coverage_masks
            unc = uncovered
            
            best_gain = -1
            best_block = -1
            best_mask = 0
            
            # Random tie-breaking:
            # If we find a gain equal to best_gain, we might swap.
            # To make it fast, we'll just pick the first one with strictly greater gain.
            # To introduce randomness, we can randomly permute the block indices at the start of the function.
            # But that's O(N log N).
            
            # Let's try: Iterate all blocks.
            for b_idx in range(n_blocks):
                m = masks[b_idx]
                # Quick check: if m is subset of already covered? No, we want intersection with uncovered.
                gain = (m & unc).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = m
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    # Using a pre-generated random number for this step?
                    # Let's just keep the first one to be fast, and rely on multiple restarts
                    # with different global seeds to explore the space.
                    pass

            if best_block == -1 or best_gain <= 0:
                break  # No improvement possible
            
            blocks.append(all_blocks[best_block])
            
            # Update uncovered
            uncovered = uncovered & ~best_mask
            remaining_uncovered -= best_gain
            
            # Early exit if no more uncovered
            if remaining_uncovered == 0:
                break
        
        # Verify completeness
        if remaining_uncovered > 0:
            return None
        
        return blocks

    def local_search_remove(blocks, rng, deadline):
        """
        Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
        Uses bitmasks.
        """
        if not blocks:
            return blocks
        
        # Convert blocks to indices
        block_to_idx = {block: i for i, block in enumerate(all_blocks)}
        block_indices = [block_to_idx[b] for b in blocks]
        
        # Compute coverage mask for the entire set of blocks
        # covered_mask = OR of all block masks
        covered_mask = 0
        for b_idx in block_indices:
            covered_mask |= coverage_masks[b_idx]
            
        # A block is redundant if (block_mask & ~covered_mask) == 0?
        # No, if we remove block i, the new covered mask is covered_mask ^ block_mask (if unique) or (covered_mask & ~block_mask) if we want to be safe?
        # Actually, if block i is the *only* one covering a specific t-subset, it's not redundant.
        # So, block i is redundant if (block_mask & ~covered_mask) == 0 is wrong.
        # It is redundant if for all t in block_mask, there exists j != i such that t in block_mask_j.
        # This is equivalent to: (block_mask & ~covered_mask) == 0 is NOT sufficient.
        # We need: (block_mask & (covered_mask_without_i)) == block_mask.
        # covered_mask_without_i is hard to compute directly with just OR.
        
        # Alternative: Count coverage.
        # But with bitsets, we can do:
        # Let S = set of blocks.
        # Block b is redundant if (mask_b & ~ (OR of all masks in S \ {b})) == 0.
        
        # Efficient way:
        # 1. Compute total_mask = OR of all masks.
        # 2. For each block b, check if mask_b is a subset of (total_mask XOR mask_b)? No.
        #    Check if (mask_b & ~total_mask) == 0? No, that just means it's covered by *something*.
        #    We need to know if it's covered by *others*.
        
        # Let's use a different approach for redundancy:
        # If we remove b, does the union change?
        # New Union = total_mask & ~mask_b? No, that removes bits ONLY covered by b.
        # If (total_mask & ~mask_b) == total_mask, then removing b doesn't change the union.
        # Wait. total_mask = OR(all).
        # If we remove b, new_union = OR(all \ {b}).
        # total_mask = new_union | mask_b.
        # If mask_b is a subset of new_union, then total_mask = new_union.
        # So b is redundant if mask_b is a subset of new_union.
        # How to check if mask_b is subset of new_union?
        # new_union = total_mask & ~ (mask_b & ~new_union)? Circular.
        
        # Let's just compute the union of all *other* blocks for each candidate? Too slow.
        
        # Heuristic: 
        # If a t-subset is covered by exactly 1 block, that block is not redundant.
        # If a t-subset is covered by >1 blocks, we might be able to remove one.
        
        # Let's try a simple iterative removal:
        # 1. Compute coverage count for each t-subset? 
        #    We can do this by iterating blocks and setting bits in a count array?
        #    Or using bitsets: 
        #    We can maintain a list of blocks.
        #    To check if block i is redundant:
        #    Let mask_i = coverage_masks[i]
        #    Let union_others = 0
        #    for j in block_indices:
        #       if j != i: union_others |= coverage_masks[j]
        #    if (mask_i & ~union_others) == 0: redundant.
        
        # This is O(N^2 * bit_ops). N=76. 76^2 = 5776. Very fast.
        
        new_block_indices = list(block_indices)
        changed = True
        
        while changed:
            changed = False
            # Shuffle to break symmetry
            rng.shuffle(new_block_indices)
            
            for i in range(len(new_block_indices) - 1, -1, -1):
                if time.time() > deadline:
                    break
                
                b_idx = new_block_indices[i]
                mask_i = coverage_masks[b_idx]
                
                # Compute union of all other blocks
                union_others = 0
                for j in range(len(new_block_indices)):
                    if j != i:
                        union_others |= coverage_masks[new_block_indices[j]]
                
                # Check if mask_i is covered by union_others
                if (mask_i & ~union_others) == 0:
                    # Redundant
                    new_block_indices.pop(i)
                    changed = True
        
        return [all_blocks[i] for i in new_block_indices]

    # Main loop: try multiple restarts
    best_blocks = None
    
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        rng_restart = random.Random(42 + restart_count)
        
        # To add randomness to greedy, we can slightly perturb the block evaluation order
        # or just rely on the fact that different seeds might lead to different tie-breaks if we implemented them.
        # Currently, my greedy is deterministic (first max). 
        # Let's add a small randomization: pick a random subset of top candidates?
        # Or, simpler: in the greedy loop, if gain == best_gain, keep the first one.
        # To make it non-deterministic, we can shuffle the block indices list once per restart.
        
        # Let's create a shuffled list of block indices for this restart
        block_order = list(range(n_blocks))
        rng_restart.shuffle(block_order)
        
        # We need to pass this order to greedy_construct or modify greedy_construct to accept it.
        # Let's modify greedy_construct to accept a block_order.
        
        # Actually, let's just do the greedy inline here or pass the order.
        # For now, let's just call the deterministic greedy. 
        # Wait, if it's deterministic, all restarts will produce the same result!
        # I MUST add randomness to the greedy step.
        
        # Revised Greedy with Randomness:
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining_uncovered = n_t
        masks = coverage_masks
        
        while remaining_uncovered > 0:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_block = -1
            best_mask = 0
            candidates = []
            
            # Iterate through shuffled block order
            for b_idx in block_order:
                m = masks[b_idx]
                gain = (m & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = m
                    candidates = [b_idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(b_idx)
            
            if best_block == -1 or best_gain <= 0:
                break
            
            # Pick randomly from candidates
            chosen_block = rng_restart.choice(candidates)
            blocks.append(all_blocks[chosen_block])
            
            # Update uncovered
            uncovered = uncovered & ~masks[chosen_block]
            remaining_uncovered -= best_gain
            
            if remaining_uncovered == 0:
                break
        
        if remaining_uncovered > 0:
            continue
        
        # Local search
        blocks = local_search_remove(blocks, rng_restart, deadline)
        
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            if time.time() > deadline - 5.0:
                break
        
        if time.time() > deadline - 15.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining_uncovered = n_t
        masks = coverage_masks
        
        while remaining_uncovered > 0:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_block = -1
            best_mask = 0
            
            for b_idx in range(n_blocks):
                m = masks[b_idx]
                gain = (m & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = m
            
            if best_block == -1 or best_gain <= 0:
                break
            
            blocks.append(all_blocks[best_block])
            uncovered = uncovered & ~best_mask
            remaining_uncovered -= best_gain
            
            if remaining_uncovered == 0:
                break
        
        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
