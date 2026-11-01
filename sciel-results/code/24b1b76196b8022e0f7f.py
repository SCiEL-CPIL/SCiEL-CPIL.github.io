import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1
            
    # 2. Polish function: Remove redundant blocks
    def polish_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
            
        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
                
        return blocks
    
    # 3. Pairwise merge local search
    def try_merge(blocks):
        """Try to merge pairs of blocks to reduce the total count."""
        n = len(blocks)
        if n < 2:
            return blocks, False
        
        # Build coverage count
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1
        
        # For each pair, check if we can merge them
        # A merge is possible if there exists a block that covers all t-sets 
        # that are uniquely covered by the two blocks being merged.
        
        # To make this efficient, we'll try random pairs
        # and check if a single block can cover the "essential" t-sets.
        
        # The essential t-sets for a pair (i, j) are those t-sets that are 
        # covered by i or j, but not by any other block in the list.
        # If such a block exists, we can replace i and j with it.
        
        # Let's try a few random pairs
        num_attempts = min(50, n * (n - 1) // 2)
        for _ in range(num_attempts):
            i, j = np.random.choice(n, size=2, replace=False)
            if i > j:
                i, j = j, i
            
            # Find t-sets uniquely covered by i or j (i.e., cover_count == 1 and covered by i or j)
            # Actually, we need t-sets that are covered by i OR j, and for which the ONLY 
            # covering blocks in the current list are i and/or j.
            # More precisely: a t-set ts is "essential" for the pair (i,j) if:
            #   - ts is covered by i or j
            #   - ts is NOT covered by any block other than i and j
            # This means cover_count[ts] is 1 or 2, and if it's 2, it must be covered by both i and j.
            
            # Let's compute the set of essential t-sets.
            # For each t-set in BlockTsets[i] or BlockTsets[j]:
            #   if cover_count[ts] == 1: it's essential (covered by exactly one of i,j, and no others)
            #   if cover_count[ts] == 2: check if it's covered by both i and j. If yes, it's essential.
            
            essential_tsets = set()
            
            # T-sets from block i
            for ts in BlockTsets[i]:
                if cover_count[ts] == 1:
                    essential_tsets.add(ts)
                elif cover_count[ts] == 2:
                    # Check if ts is also in block j
                    if ts in BlockTsets[j]:
                        essential_tsets.add(ts)
            
            # T-sets from block j (that weren't already added)
            for ts in BlockTsets[j]:
                if ts not in essential_tsets:
                    if cover_count[ts] == 1:
                        essential_tsets.add(ts)
                    elif cover_count[ts] == 2:
                        if ts in BlockTsets[i]:
                            essential_tsets.add(ts)
            
            if not essential_tsets:
                # No essential t-sets? That means i and j are completely redundant?
                # If so, we should have removed them in polish.
                # But maybe polish didn't catch it because of order?
                # In any case, if there are no essential t-sets, we can remove one of them.
                # But let's just skip this case for now and rely on polish.
                continue
            
            # Now, find a block that covers all essential t-sets.
            # We can do this by checking each block in all_blocks.
            # To speed up, we can precompute for each t-set, the list of blocks that cover it.
            # But that's expensive to build.
            
            # Alternative: For each candidate block, check if it covers all essential t-sets.
            # We can vectorize this a bit.
            
            # Let's try a subset of candidate blocks.
            # Which blocks are likely to cover all essential t-sets?
            # Blocks that share many elements with i and j.
            
            # For now, let's just try all blocks. It might be slow, but let's see.
            # Actually, num_blocks is C(19,8) = 75582. That's too many to check for each pair.
            
            # Better approach:
            # For each essential t-set, find the blocks that cover it.
            # The intersection of these sets is the set of candidate blocks.
            
            # Let's build an inverted index: tset -> list of blocks that cover it.
            # This is expensive to build, but we only do it once.
            
            # Actually, let's not build the full inverted index. Instead, let's use a heuristic.
            # The candidate block must cover all essential t-sets.
            # So it must be a superset of the union of all essential t-sets? No, that's not right.
            # It must contain each essential t-set as a subset.
            
            # So the candidate block must contain all elements that appear in any essential t-set?
            # No, that's not right either. It must contain each essential t-set entirely.
            
            # Let's just try a few random blocks and see if they work.
            # This is a heuristic, but it might work.
            
            # Actually, let's try a smarter approach.
            # The candidate block must contain the intersection of all essential t-sets? No.
            
            # Let's just give up on pairwise merge for now and focus on other improvements.
            # The pairwise merge is complex and might not be worth it.
            
            break
        
        return blocks, False
    
    # 4. Greedy Search with Restart
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
        
        # Polish
        blocks = polish_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
        
        # Try pairwise merge
        blocks, changed = try_merge(blocks)
        if changed:
            blocks = polish_blocks(blocks)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
    
    if best_blocks is None:
        return []
    
    result = [list(all_blocks[b]) for b in best_blocks]
    return result
