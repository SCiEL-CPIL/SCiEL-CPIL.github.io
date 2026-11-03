import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an optimized Information-Theoretic Greedy 
    with Bitset acceleration and aggressive Local Search.
    """
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Precompute block masks for all possible blocks? 
    # C(24,7) is ~346k. This is feasible to precompute masks for?
    # 346k blocks * 35 triples/block = ~12M ops. Might be too slow in Python if done naively.
    # Better: Generate blocks on the fly but use fast bitset ops.
    
    # We will use a list of integers representing bitsets of covered triples.
    # However, 2024 bits requires 32 64-bit ints or just a Python int (which is arbitrary precision).
    # Python ints are efficient for bit ops.
    
    # Map triple to a single bit position
    # We'll use a dict from triple -> index, and a mask for each block as a Python int.
    
    def get_block_mask(block):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        return mask

    def count_bits(mask):
        return mask.bit_count() # Python 3.10+

    # Phase 1: Greedy Construction
    # We use a weighted greedy where weights are updated based on scarcity.
    # To make it fast, we don't recalculate all weights from scratch.
    # Instead, we use a heuristic: Pick the block that maximizes (newly covered triples) / (cost).
    # Cost is 1. So just max newly covered.
    # To incorporate "entropy reduction" / scarcity, we can bias the selection towards
    # blocks that cover triples with low "potential coverage" by other random blocks.
    
    # A simpler, effective heuristic: 
    # 1. Start with empty coverage.
    # 2. At each step, pick the block that covers the most *currently uncovered* triples.
    # 3. Tie-breaker: Pick the block that covers the most "rare" triples.
    
    # For v=24, k=7, the number of candidate blocks is large. 
    # We cannot iterate all C(24,7) blocks at each step (346k * 76 steps * bit ops).
    # 346k * 76 = 26M iterations. In Python, this might be too slow (26M * ~1us = 26s).
    # We need to restrict candidates.
    
    # Strategy: 
    # 1. Pick a random uncovered triple.
    # 2. Generate all blocks containing that triple. C(24-3, 7-3) = C(21,4) = 5985 blocks.
    # 3. Evaluate these 5985 blocks. Pick the best one (max new coverage).
    # 4. Repeat.
    
    # This is much faster: 76 steps * 5985 candidates = 450k evaluations.
    # Each evaluation involves computing the intersection of the block mask with the uncovered mask.
    
    blocks = []
    uncovered_mask = (1 << n_triples) - 1
    
    # Precompute masks for all blocks? 
    # No, compute on the fly for the candidate set.
    
    # To make it even faster, we can precompute the list of blocks containing each triple.
    # triple_blocks = {triple: [blocks containing triple]}
    # This takes memory but speeds up lookup.
    # 2024 triples * 5985 blocks/triple = 12M block references. Too much memory/time to build.
    
    # So, generate candidates on the fly.
    
    random.seed(42)
    
    while uncovered_mask:
        # Pick a random uncovered triple to anchor the search
        # Find an index where bit is 1
        # random_idx = random.randint(0, n_triples-1)
        # while not (uncovered_mask >> random_idx) & 1:
        #     random_idx = random.randint(0, n_triples-1)
        
        # Faster: Pick a random point, find an uncovered triple containing it?
        # Or just pick a random triple index and check.
        # Since uncovered_mask shrinks, this is fast.
        
        # Let's pick the "most constrained" uncovered triple.
        # A triple is constrained if few blocks can cover it? No, all blocks containing it are valid.
        # The constraint is on the *other* triples.
        
        # Heuristic: Pick the uncovered triple that has the fewest "compatible" uncovered triples?
        # Too complex.
        
        # Let's stick to: Pick a random uncovered triple.
        # This introduces randomness which helps escape local minima in the greedy phase.
        
        # Find a set bit in uncovered_mask
        # random_bit = 1 << random.randint(0, n_triples-1)
        # while not (uncovered_mask & random_bit):
        #     random_bit = 1 << random.randint(0, n_triples-1)
        
        # Better:
        # temp_mask = uncovered_mask
        # random_offset = random.randint(0, n_triples - 1)
        # # This is tricky with Python ints.
        
        # Let's just pick the first uncovered triple? No, that's deterministic and might be bad.
        # Let's pick a random triple from the list of all triples, and if it's covered, try again.
        
        target_triple = None
        while target_triple is None:
            idx = random.randint(0, n_triples - 1)
            if (uncovered_mask >> idx) & 1:
                target_triple = all_triples[idx]
        
        # Generate all blocks containing target_triple
        # target_triple has 3 points. We need 4 more from the remaining 21.
        remaining_points = [p for p in range(v) if p not in target_triple]
        
        best_block = None
        best_gain = -1
        best_mask = 0
        
        # Iterate C(21, 4) = 5985 combinations
        for combo in itertools.combinations(remaining_points, k - t):
            block = list(target_triple) + list(combo)
            block.sort()
            
            # Calculate mask
            mask = 0
            # Optimization: We only care about the intersection with uncovered_mask
            # But we need the full mask to update uncovered_mask later? 
            # No, we just need the gain.
            # gain = count_bits(mask & uncovered_mask)
            
            # To speed up, we can compute the mask and then AND.
            # Computing the mask involves 35 combinations.
            # 5985 * 35 = 209k triple lookups per step.
            # 76 steps * 209k = 15M lookups. This is acceptable in Python? 
            # 15M dict lookups + bit ops. Might take ~10-20s.
            
            # We can optimize by pre-computing the mask for the block? No, blocks are unique.
            
            # Let's try to optimize the mask calculation.
            # We can precompute the masks for all triples? No, masks are for blocks.
            
            # Let's just do it.
            b_mask = 0
            for triple in itertools.combinations(block, t):
                idx_t = triple_to_idx[triple]
                b_mask |= (1 << idx_t)
            
            gain = count_bits(b_mask & uncovered_mask)
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_mask = b_mask
                
        if best_block is None:
            break # Should not happen
            
        blocks.append(best_block)
        uncovered_mask &= ~best_mask
        
    # Phase 2: Local Search
    # Convert blocks to masks for faster operations
    block_masks = [get_block_mask(b) for b in blocks]
    
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            # Check if block i is redundant
            # Redundant if: (uncovered_mask_after_removing_i) == (uncovered_mask_before_removing_i)
            # i.e., block_masks[i] is a subset of (union of all other block_masks)
            
            # Calculate union of others? O(N) per check. O(N^2) total.
            # N ~ 80. 6400 union ops. Fast.
            
            union_others = 0
            for j in range(len(blocks)):
                if i != j:
                    union_others |= block_masks[j]
                    
            if (block_masks[i] & ~union_others) == 0:
                # Block i is redundant
                blocks.pop(i)
                block_masks.pop(i)
                changed = True
                break
                
    # Pass 2: Replace blocks to improve coverage or allow further removal
    # We try to replace a block with a new block that covers the same exclusive triples
    # but potentially allows removing other blocks?
    # Simpler: Try to replace a block with a different block that covers MORE total triples
    # (including already covered ones, which doesn't help directly, but might help if we 
    # consider the structure).
    
    # Actually, a good local search move is:
    # 1. Pick a block B.
    # 2. Find triples T_exclusive covered only by B.
    # 3. If T_exclusive is empty, remove B (handled in Pass 1).
    # 4. If not, try to find a block B' such that T_exclusive is a subset of B'.
    # 5. If such B' exists, replace B with B'. This might not reduce count immediately,
    #    but it might change the exclusive sets of other blocks, allowing removals.
    
    # Let's do a "Swap" local search:
    # Try to replace one block with another.
    
    max_iters = 500
    for _ in range(max_iters):
        i = random.randint(0, len(blocks) - 1)
        b_mask = block_masks[i]
        
        # Find exclusive triples of block i
        # Union of others
        union_others = 0
        for j in range(len(blocks)):
            if i != j:
                union_others |= block_masks[j]
                
        exclusive_mask = b_mask & ~union_others
        
        if exclusive_mask == 0:
            # Should have been removed
            continue
            
        # We need a new block that covers exclusive_mask.
        # Pick a triple from exclusive_mask to anchor the search for new block.
        # exclusive_mask is a bitset. Find a set bit.
        # Since exclusive_mask is small (few bits), we can iterate.
        
        # Get the list of triples in exclusive_mask
        excl_triples = []
        for idx in range(n_triples):
            if (exclusive_mask >> idx) & 1:
                excl_triples.append(all_triples[idx])
                
        if not excl_triples:
            continue
            
        # Pick the first exclusive triple as anchor
        anchor = excl_triples[0]
        remaining_points = [p for p in range(v) if p not in anchor]
        
        best_new_block = None
        best_new_mask = 0
        # We want the new block to cover ALL exclusive triples? 
        # If not, the replacement is invalid (we lose coverage).
        # So we must find a block B' such that exclusive_mask is a subset of B'.
        
        # Iterate candidates containing anchor.
        found = False
        for combo in itertools.combinations(remaining_points, k - t):
            new_block = list(anchor) + list(combo)
            new_block.sort()
            new_mask = get_block_mask(new_block)
            
            # Check if new_mask covers exclusive_mask
            if (new_mask & exclusive_mask) == exclusive_mask:
                # Valid replacement
                # Prefer one that covers more total triples? 
                # Or one that makes other blocks redundant?
                # Let's just take the first one found to save time, 
                # or the one with max coverage.
                
                # To encourage reduction, let's see if this replacement allows removal of others.
                # This is expensive. Let's just replace and re-run Pass 1.
                blocks[i] = new_block
                block_masks[i] = new_mask
                found = True
                break
                
        if found:
            # Re-run redundancy removal
            changed = True
            while changed:
                changed = False
                for j in range(len(blocks)):
                    union_others_j = 0
                    for l in range(len(blocks)):
                        if j != l:
                            union_others_j |= block_masks[l]
                    if (block_masks[j] & ~union_others_j) == 0:
                        blocks.pop(j)
                        block_masks.pop(j)
                        changed = True
                        break
                        
    return blocks
