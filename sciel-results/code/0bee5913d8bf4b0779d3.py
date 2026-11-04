import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Pre-compute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Bitset representation for triples in a block
    # Since num_triples (2024) > 64, we use Python's arbitrary precision integers as bitsets.
    # This is much faster than lists for intersection/counting if we use bit_count().
    
    def get_block_bitset(block):
        mask = 0
        for tri in itertools.combinations(block, t):
            idx = triple_to_idx[tri]
            mask |= (1 << idx)
        return mask

    # 1. Fast Greedy Construction
    # We use a bitset of uncovered triples to quickly find the best next block.
    def greedy_construct():
        uncovered_mask = (1 << num_triples) - 1  # All triples uncovered
        blocks = []
        
        while uncovered_mask:
            # Find a block that maximizes coverage of uncovered triples
            # To speed this up, we don't check all C(v,k) blocks.
            # We pick a random uncovered triple and build a block around it,
            # or we sample a few random blocks and pick the best.
            # Given the size, sampling is safer.
            
            best_gain = -1
            best_block = None
            best_mask = 0
            
            # Sample 1000 random blocks
            for _ in range(1000):
                b = tuple(sorted(random.sample(range(v), k)))
                mask = get_block_bitset(b)
                gain = (mask & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    best_mask = mask
                if gain == num_triples: # Perfect block (unlikely)
                    break
            
            if best_block is None:
                # Fallback
                b = tuple(sorted(random.sample(range(v), k)))
                best_block = b
                best_mask = get_block_bitset(b)
                
            blocks.append(best_block)
            uncovered_mask &= ~best_mask
            
        return blocks

    current_blocks = greedy_construct()
    best_blocks = current_blocks[:]
    best_len = len(current_blocks)
    
    # State: counts array for each triple
    counts = [0] * num_triples
    for b in current_blocks:
        for tri_idx in (triple_to_idx[tri] for tri in itertools.combinations(b, t)):
            counts[tri_idx] += 1
    
    # Pre-compute bitsets for current blocks to speed up swap evaluation
    # block_masks[i] is the bitset for current_blocks[i]
    block_masks = [get_block_bitset(b) for b in current_blocks]
    
    # Helper to get list of triple indices from a mask (only for small masks or debugging)
    # For optimization, we mostly use counts.
    
    def get_block_indices(block):
        return [triple_to_idx[tri] for tri in itertools.combinations(block, t)]

    # Local Search: Directed Replacement
    # Strategy: Find triples with count == 1 (critical).
    # Pick one critical triple. Find a block containing it.
    # Try to replace that block with a new block that:
    # 1. Contains the critical triple.
    # 2. Covers as many other count==1 triples as possible.
    # 3. Minimizes the number of count==1 triples that would become uncovered (i.e., triples in old block not in new block that have count==1).
    
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Pre-compute list of all blocks? No, too many.
    # We will generate candidates on the fly.
    
    steps = 0
    while time.time() < deadline:
        # Identify critical triples (count == 1)
        critical_triples = [i for i, c in enumerate(counts) if c == 1]
        
        if not critical_triples:
            # No critical triples, try to remove redundant blocks
            # Find a block that is fully redundant (all its triples have count > 1)
            redundant_idx = -1
            for i, mask in enumerate(block_masks):
                # Check if any triple in this block has count == 1
                # We can iterate through the set bits of mask
                m = mask
                is_redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                if is_redundant:
                    redundant_idx = i
                    break
            
            if redundant_idx != -1:
                # Remove it
                old_block = current_blocks[redundant_idx]
                old_mask = block_masks[redundant_idx]
                
                # Update counts
                m = old_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                
                current_blocks.pop(redundant_idx)
                block_masks.pop(redundant_idx)
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
            else:
                # No redundant blocks, force a swap to escape local minimum?
                # Or just break if stuck? Let's try a random swap.
                pass
        
        # Pick a random critical triple to focus on
        if critical_triples:
            target_tri_idx = random.choice(critical_triples)
            target_tri = all_triples[target_tri_idx]
            
            # Find a block that contains this triple
            # We can search current_blocks
            containing_block_idx = -1
            for i, b in enumerate(current_blocks):
                if target_tri[0] in b and target_tri[1] in b and target_tri[2] in b:
                    containing_block_idx = i
                    break
            
            if containing_block_idx == -1:
                # Should not happen if valid, but if we allowed invalid states...
                # Our greedy ensures validity, and swaps maintain validity if we check.
                # Actually, if we do directed swaps, we must ensure the new block covers the critical triple.
                continue

            old_block = current_blocks[containing_block_idx]
            old_mask = block_masks[containing_block_idx]
            
            # Generate candidate new blocks that contain target_tri
            # We need to choose k-3 more elements from v-3
            # To make it efficient, we sample a few candidates
            best_candidate = None
            best_score = -1
            best_new_mask = 0
            
            # Sample 200 candidates
            for _ in range(200):
                # Choose 4 random elements from the rest
                remaining = [x for x in range(v) if x not in target_tri]
                extra = random.sample(remaining, k - t)
                new_block = tuple(sorted(list(target_tri) + extra))
                
                if new_block == old_block:
                    continue
                    
                new_mask = get_block_bitset(new_block)
                
                # Calculate Score
                # Gain: Number of critical triples in new_block that are NOT in old_block
                # Loss: Number of critical triples in old_block that are NOT in new_block
                # Net change in critical count = Loss - Gain
                # We want to minimize this (or maximize Gain - Loss)
                
                # Triples in new_block but not in old_block
                gain_mask = new_mask & ~old_mask
                # Triples in old_block but not in new_block
                loss_mask = old_mask & ~new_mask
                
                # Count how many of these are critical (count == 1)
                # Note: A triple is critical if counts[tri_idx] == 1.
                # Since the old block is the ONLY block covering these critical triples (by definition of critical),
                # any critical triple in loss_mask will become uncovered (count 0) -> BAD.
                # Any critical triple in gain_mask will become covered by new block (count 1) -> GOOD (it was already covered by old, so count becomes 2? No.)
                
                # Wait, logic check:
                # Old block covers T. counts[T] = 1.
                # If T is in old_block and NOT in new_block: counts[T] becomes 0. (Uncovered!)
                # If T is in new_block and NOT in old_block: 
                #    If T was critical (count 1), it must be covered by some other block? No, critical means count 1.
                #    If T is in new_block but not old, and T was critical, it was covered by old? No, T is not in old.
                #    So T must be covered by some OTHER block? But count is 1.
                #    Contradiction. If T is critical, it is covered by exactly one block.
                #    If we are replacing that block, T must be in the old block.
                #    So any critical triple MUST be in the old block.
                #    Therefore, gain_mask cannot contain any critical triples?
                #    Unless... the "critical" status is calculated BEFORE the swap.
                #    If T is in new_block, and T is not in old_block, and T was critical (count 1),
                #    then T was covered by some block B. If B != old, then count(T) >= 1.
                #    But we are replacing old. If old doesn't contain T, count(T) stays 1 (covered by B).
                #    So adding new_block makes count(T) 2.
                #    So Gain for critical triples from new_block \ old_block is 0?
                #    No, it increases redundancy, which is good for stability, but doesn't reduce the count of critical triples directly?
                #    
                #    Let's re-evaluate the objective.
                #    We want to reduce the total number of blocks.
                #    The bottleneck is that we can't remove blocks because they have critical triples.
                #    To remove a block, we need to ensure all its triples are covered by others.
                #    So we need to "shift" the coverage of critical triples to other blocks.
                #    
                #    If we replace Old with New:
                #    1. Triples in Old \ New: Lose coverage from Old.
                #       If they were critical (count 1), they become uncovered (count 0). INVALID.
                #       So we MUST ensure Old \ New contains NO critical triples.
                #       i.e., loss_mask & critical_mask == 0.
                #       
                #    2. Triples in New \ Old: Gain coverage from New.
                #       If they were critical (count 1), they become count 2.
                #       This "fixes" them (they are no longer critical).
                #       This is GOOD.
                #       
                #    3. Triples in Old & New: Count stays same (Old removed, New added).
                #    
                #    So the condition for a VALID swap is:
                #    (old_mask & ~new_mask) & critical_mask == 0
                #    
                #    And the BENEFIT is:
                #    Number of critical triples in (new_mask & ~old_mask).
                #    These triples will no longer be critical after the swap.
            
                critical_mask = 0
                # We can precompute critical_mask as a bitset for faster ops?
                # Or just check on the fly.
                # Let's compute critical_mask once per iteration outside the candidate loop.
                
            # Recompute critical_mask for the current state
            critical_mask = 0
            for i, c in enumerate(counts):
                if c == 1:
                    critical_mask |= (1 << i)
            
            # Now loop candidates again with proper logic
            best_candidate = None
            best_benefit = -1
            best_new_mask = 0
            
            for _ in range(200):
                remaining = [x for x in range(v) if x not in target_tri]
                extra = random.sample(remaining, k - t)
                new_block = tuple(sorted(list(target_tri) + extra))
                
                if new_block == old_block:
                    continue
                    
                new_mask = get_block_bitset(new_block)
                
                # Check validity: No critical triple is lost
                lost_critical = (old_mask & ~new_mask) & critical_mask
                if lost_critical:
                    continue
                
                # Calculate benefit: How many critical triples are gained (become non-critical)
                # These are critical triples in new_block that were NOT in old_block.
                # Wait, if a triple is critical, it is covered by exactly one block.
                # If it's in new_block and not old_block, and it's critical,
                # it must be covered by some OTHER block.
                # So its count is currently 1 (from that other block).
                # After adding new_block, its count becomes 2.
                # So it is no longer critical.
                gained_critical = (new_mask & ~old_mask) & critical_mask
                benefit = gained_critical.bit_count()
                
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_candidate = new_block
                    best_new_mask = new_mask
                    
                # If benefit is high, we can stop early?
                # Max possible benefit is small.
                
            if best_candidate:
                # Apply Swap
                # 1. Update counts
                # Remove old
                m = old_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                # Add new
                m = best_new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
                
                # 2. Update lists
                current_blocks[containing_block_idx] = best_candidate
                block_masks[containing_block_idx] = best_new_mask
                
                # Track best
                # We didn't change the number of blocks, so best_len doesn't change.
                # But we improved the "flexibility" (fewer critical triples).
                
        # Periodic cleanup: Remove redundant blocks
        # This is the step that actually reduces the count.
        # We do this frequently to harvest the fruits of the swaps.
        if steps % 10 == 0:
            changed = True
            while changed:
                changed = False
                for i in range(len(current_blocks)):
                    mask = block_masks[i]
                    is_redundant = True
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 1:
                            is_redundant = False
                            break
                        m ^= lsb
                    if is_redundant:
                        # Remove
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        current_blocks.pop(i)
                        block_masks.pop(i)
                        changed = True
                        break # Restart loop
                
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks[:]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        steps += 1

    # Final cleanup
    final_blocks = []
    final_counts = [0] * num_triples
    for b in best_blocks:
        for tri_idx in (triple_to_idx[tri] for tri in itertools.combinations(b, t)):
            final_counts[tri_idx] += 1
            
    for b in best_blocks:
        b_indices = get_block_indices(b)
        can_remove = True
        for tri_idx in b_indices:
            if final_counts[tri_idx] == 1:
                can_remove = False
                break
        if not can_remove:
            final_blocks.append(b)
            for tri_idx in b_indices:
                final_counts[tri_idx] -= 1
        else:
            # It was redundant, don't add it
            pass
            
    # Verify
    if min(final_counts) < 1:
        # Fallback to original best_blocks if cleanup broke something (shouldn't happen)
        return [list(b) for b in best_blocks]
        
    return [list(b) for b in final_blocks]
