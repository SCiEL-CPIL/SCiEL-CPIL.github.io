import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # For fast mask generation, we can't precompute all blocks.
    # But we can precompute the mask contribution of a triplet? No, mask is per block.
    # We will compute block masks on the fly.
    
    def get_block_mask(block):
        mask = 0
        # t=3, k=7 -> 35 triplets
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Helper to enumerate blocks containing a set of points
    def blocks_containing(points):
        points = frozenset(points)
        if len(points) > k:
            return
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points | set(comb)))

    # Greedy Construction
    def greedy_construct(seed_val):
        random.seed(seed_val)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet to guide the search
            # Find index of a set bit in uncovered
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = all_triplets[idx]
            
            # Find best block containing start_triplet
            # Heuristic: maximize newly covered triplets
            best_block = None
            best_gain = -1
            
            # To speed up, we can sample or iterate all. 
            # Number of blocks containing a specific triplet is C(24-3, 7-3) = C(21,4) = 5985.
            # Iterating 5985 blocks and computing masks (35 triplets each) is ~200k ops.
            # This is fast enough for a few hundred iterations.
            
            candidates = list(blocks_containing(start_triplet))
            random.shuffle(candidates)
            
            for block in candidates:
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    # Early exit if we found a perfect block (covers all uncovered? unlikely)
                    if gain == uncovered.bit_count():
                        break
            
            blocks.append(best_block)
            uncovered &= ~get_block_mask(best_block)
            
        return blocks

    # Local Search Optimizer
    def optimize(blocks, seed_val):
        random.seed(seed_val)
        
        # Maintain counts for redundancy checks
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        # Cache block masks to avoid recomputation
        block_masks = [get_block_mask(b) for b in blocks]
        
        def update_mask(idx, new_block):
            block_masks[idx] = get_block_mask(new_block)
            
        def is_valid():
            return np.min(counts) > 0
            
        max_iters = 5000
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove block i
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks.pop(i)
                        block_masks.pop(i)
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                # Randomly pick a pair
                if n >= 2:
                    i, j = random.sample(range(n), 2)
                    if i > j: i, j = j, i
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    # Compute union of masks
                    mask1 = block_masks[i]
                    mask2 = block_masks[j]
                    union_mask = mask1 | mask2
                    
                    # Find triplets uniquely covered by b1 or b2
                    # These are triplets in union_mask where count == 1
                    # Note: If a triplet is in both, count >= 2, so it's not unique.
                    # So we just need to cover all triplets in union_mask that have count 1.
                    
                    needed_triplets = []
                    # Iterate through set bits in union_mask
                    temp_mask = union_mask
                    while temp_mask:
                        lsb = temp_mask & -temp_mask
                        tid = lsb.bit_length() - 1
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                        temp_mask ^= lsb
                        
                    if not needed_triplets:
                        # Both are redundant (should have been caught by step 1, but maybe interaction?)
                        # Remove both
                        for triplet in itertools.combinations(b2, t):
                            counts[triplet_to_id[triplet]] -= 1
                        for triplet in itertools.combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks.pop(j)
                        block_masks.pop(j)
                        blocks.pop(i)
                        block_masks.pop(i)
                        improved = True
                        break
                        
                    # Find points in needed_triplets
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = all_triplets[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    # Try to find a block covering points_needed
                    found = False
                    # Limit search for speed
                    count = 0
                    for new_block in blocks_containing(points_needed):
                        count += 1
                        if count > 1000: break # Heuristic limit
                        
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        new_mask = get_block_mask(new_block)
                        
                        # Check if new_mask covers all needed_triplets
                        # needed_mask = OR of (1 << tid) for tid in needed_triplets
                        # We can just check intersection
                        # But we already know needed_triplets are in union_mask.
                        # We need new_mask to contain all bits in needed_triplets.
                        
                        # Construct needed_mask
                        needed_mask = 0
                        for tid in needed_triplets:
                            needed_mask |= (1 << tid)
                            
                        if (new_mask & needed_mask) == needed_mask:
                            # Perform swap
                            # Remove i and j
                            for triplet in itertools.combinations(b2, t):
                                counts[triplet_to_id[triplet]] -= 1
                            for triplet in itertools.combinations(b1, t):
                                counts[triplet_to_id[triplet]] -= 1
                            blocks.pop(j)
                            block_masks.pop(j)
                            blocks.pop(i)
                            block_masks.pop(i)
                            
                            # Add new_block
                            blocks.append(new_block)
                            block_masks.append(new_mask)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                                
                            improved = True
                            found = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 1-for-1
            if not improved:
                # Pick random block
                i = random.randint(0, n-1)
                b1 = blocks[i]
                mask1 = block_masks[i]
                
                # Find unique triplets in b1
                needed_triplets = []
                temp_mask = mask1
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    tid = lsb.bit_length() - 1
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                    temp_mask ^= lsb
                    
                if not needed_triplets:
                    # Redundant
                    for triplet in itertools.combinations(b1, t):
                        counts[triplet_to_id[triplet]] -= 1
                    blocks.pop(i)
                    block_masks.pop(i)
                    improved = True
                    break
                    
                points_needed = set()
                for tid in needed_triplets:
                    trip = all_triplets[tid]
                    points_needed.update(trip)
                    
                if len(points_needed) > k:
                    continue
                    
                found = False
                count = 0
                for new_block in blocks_containing(points_needed):
                    count += 1
                    if count > 1000: break
                    
                    if set(new_block) == set(b1):
                        continue
                        
                    new_mask = get_block_mask(new_block)
                    
                    needed_mask = 0
                    for tid in needed_triplets:
                        needed_mask |= (1 << tid)
                        
                    if (new_mask & needed_mask) == needed_mask:
                        # Replace
                        for triplet in itertools.combinations(b1, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks[i] = new_block
                        block_masks[i] = new_mask
                        for triplet in itertools.combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        found = True
                        break
                        
                if improved:
                    break
                    
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    # We have ~145s.
    # Greedy is fast. Optimization is the bottleneck.
    # Let's do 10-20 restarts.
    
    for seed in range(20):
        if time.time() > deadline - 10:
            break
            
        init_blocks = greedy_construct(seed)
        opt_blocks = optimize(init_blocks, seed)
        
        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_construct(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
