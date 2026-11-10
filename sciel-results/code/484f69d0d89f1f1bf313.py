# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using bitset-accelerated greedy
    with randomized restarts and redundant-block removal.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute a Python int bitmask of which t-subsets it covers
    # C(24,3) = 2024 bits, so each mask is a ~253-byte int
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        block_masks.append(mask)

    # For each t-subset, list of block indices that contain it
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx, mask in enumerate(block_masks):
        # Extract set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            blocks_containing[idx].append(b_idx)
            m ^= lsb

    def greedy_bitset(rng, deadline):
        """
        Bitset-accelerated greedy set cover.
        uncovered is a Python int bitmask of uncovered t-subsets.
        """
        # Initialize: all t-subsets uncovered
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            # Find the block that maximizes popcount(block_mask & uncovered)
            best_block = -1
            best_gain = 0
            best_mask = 0

            # Randomize order to break ties
            order = list(range(n_blocks))
            rng.shuffle(order)

            for b_idx in order:
                if time.time() > deadline:
                    break
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_mask = block_masks[b_idx]
                elif gain == best_gain and gain > 0 and rng.random() < 0.3:
                    best_block = b_idx
                    best_mask = block_masks[b_idx]

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~best_mask
            remaining = uncovered.bit_count()

        if remaining > 0:
            return None
        return blocks

    def remove_redundant(blocks, deadline):
        """
        Remove redundant blocks: a block is redundant if every t-subset it
        covers is also covered by at least one other block.
        """
        if not blocks:
            return blocks

        # Map block tuples to their index in all_blocks
        block_set = set(blocks)
        n = len(blocks)

        # Compute coverage count per t-subset using bitmasks
        # cov_count[i] = number of blocks covering t-subset i
        cov_count = [0] * n_t
        for b in blocks:
            b_idx = all_blocks.index(b)
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb

        changed = True
        while changed:
            changed = False
            # Try removing blocks in random order
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    return blocks

                b = blocks[i]
                b_idx = all_blocks.index(b)
                mask = block_masks[b_idx]

                # Check if any t-subset is uniquely covered by this block
                m = mask
                redundant = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb

                if redundant:
                    blocks.pop(i)
                    # Update coverage counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    changed = True

        return blocks

    def local_search(blocks, deadline):
        """
        Local search: try to replace each block with another that covers 
        more unique t-subsets, or remove blocks that become redundant after
        other removals. Also try swapping pairs of blocks.
        """
        if not blocks:
            return blocks
        
        n = len(blocks)
        if n == 0:
            return blocks
        
        # Precompute masks for current blocks
        block_masks_local = []
        for b in blocks:
            b_idx = all_blocks.index(b)
            block_masks_local.append(block_masks[b_idx])
        
        # Compute coverage counts
        cov_count = [0] * n_t
        for mask in block_masks_local:
            m = mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lsb
        
        # Try to improve by replacing blocks
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() > deadline:
                break
            
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                # Try removing block i
                mask_i = block_masks_local[i]
                m = mask_i
                can_remove = True
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if can_remove:
                    # Remove block i
                    blocks.pop(i)
                    block_masks_local.pop(i)
                    m = mask_i
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    improved = True
                    continue
                
                # Try replacing block i with a better one
                # Find t-subsets uniquely covered by block i
                unique_to_i = []
                m = mask_i
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    if cov_count[idx] == 1:
                        unique_to_i.append(idx)
                    m ^= lsb
                
                if not unique_to_i:
                    continue
                
                # Look for a block that covers all unique_to_i and has more total coverage
                best_replace = -1
                best_score = -1
                unique_mask = 0
                for idx in unique_to_i:
                    unique_mask |= (1 << idx)
                
                # Sample some candidate blocks to keep it fast
                candidates = random.sample(range(n_blocks), min(500, n_blocks))
                for b_idx in candidates:
                    if b_idx == all_blocks.index(blocks[i]):
                        continue
                    cand_mask = block_masks[b_idx]
                    # Must cover all unique t-subsets
                    if (cand_mask & unique_mask) != unique_mask:
                        continue
                    # Score: number of new t-subsets covered
                    new_coverage = (cand_mask & ~mask_i).bit_count()
                    if new_coverage > best_score:
                        best_score = new_coverage
                        best_replace = b_idx
                
                if best_replace >= 0 and best_score > 0:
                    # Replace block i with best_replace
                    old_mask = block_masks_local[i]
                    new_mask = block_masks[best_replace]
                    
                    # Update coverage counts
                    m = old_mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] -= 1
                        m ^= lsb
                    
                    m = new_mask
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] += 1
                        m ^= lsb
                    
                    blocks[i] = all_blocks[best_replace]
                    block_masks_local[i] = new_mask
                    improved = True
        
        return blocks

    # Main loop: multiple randomized restarts
    best_blocks = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1
        rng = random.Random(42 + restart_count)

        blocks = greedy_bitset(rng, deadline)

        if blocks is not None:
            blocks = remove_redundant(blocks, deadline)
            blocks = local_search(blocks, deadline)
            blocks = remove_redundant(blocks, deadline)  # Clean up after local search

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
                # If we've found a good solution, spend more time refining
                if best_len <= 78:
                    # Do additional local search passes
                    for _ in range(3):
                        if time.time() > deadline - 20.0:
                            break
                        blocks = local_search(blocks, deadline)
                        blocks = remove_redundant(blocks, deadline)
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = blocks

        # Allow more restarts if we're doing well time-wise
        if time.time() > deadline - 20.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_blocks is None:
        uncovered = (1 << n_t) - 1
        blocks = []
        remaining = n_t

        while remaining > 0:
            if time.time() > deadline:
                break

            best_block = -1
            best_gain = 0
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block == -1 or best_gain <= 0:
                break

            blocks.append(all_blocks[best_block])
            uncovered &= ~block_masks[best_block]
            remaining = uncovered.bit_count()

        if remaining > 0:
            # Should not happen
            return blocks

        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
