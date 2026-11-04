import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Map triplet id to the triplet itself for reverse lookup if needed
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute masks for triplets? No, we compute block masks on the fly.
    # But we need to quickly get the mask of a block.
    # block_mask(block): OR of (1 << id) for all t-subsets of block.
    
    def get_block_mask(block):
        mask = 0
        # For t=3, k=7, 35 triplets
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute all possible blocks? No, too many.
    # But we need to enumerate blocks containing a set of points.
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        # Need k - len(points) more points from remaining
        need = k - len(points)
        # Generate combinations
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Greedy Construction with ratio-based selection
    def greedy_construct():
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet to guide the search
            # Find index of a set bit in uncovered
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet using ratio: gain / cost
            # where gain = newly covered triplets, cost = total triplets in block
            best_block = None
            best_ratio = -1
            best_mask = 0
            
            # Enumerate all blocks containing start_triplet
            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                cost = b_mask.bit_count()  # total triplets in this block
                ratio = gain / max(cost, 1)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_block = block
                    best_mask = b_mask
            
            blocks.append(best_block)
            uncovered &= ~best_mask
            
        return blocks

    # Local Search Optimizer
    def optimize(blocks):
        # Maintain counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        block_masks = {}
        
        def update_counts(block, delta):
            mask = get_block_mask(block)
            # Update counts for bits set in mask
            # This is slow if done bit-by-bit.
            # Better to just recompute counts from scratch?
            # N_blocks ~ 70. N_triplets ~ 2000.
            # 70 * 35 = 2450 updates. Fast.
            pass
        
        # Let's use a simpler representation: list of blocks and a function to get coverage
        # But for speed, we'll maintain `counts` array.
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        def is_valid():
            return np.min(counts) > 0
        
        def get_unique_mask(block_idx, b_list):
            # Returns mask of triplets covered ONLY by b_list[block_idx]
            b = b_list[block_idx]
            b_mask = get_block_mask(b)
            # Compute mask of others
            others_mask = 0
            for i, ob in enumerate(b_list):
                if i != block_idx:
                    others_mask |= get_block_mask(ob)
            return b_mask & ~others_mask

        # Optimization Loop
        max_iters = 10000
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            n = len(blocks)
            
            # 1. Try Remove Redundant
            for i in range(n):
                if not improved:
                    # Check if blocks[i] is redundant
                    # If counts[t] > 1 for all t in blocks[i]
                    b = blocks[i]
                    redundant = True
                    for triplet in itertools.combinations(b, t):
                        if counts[triplet_to_id[triplet]] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove it
                        blocks.pop(i)
                        # Update counts
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break # Restart loop as indices changed
                        
            # 2. Try 2-for-1
            if not improved:
                # Randomly pick a pair to avoid getting stuck in deterministic order
                # Or try all pairs? N is small (~70). 70*69/2 = 2415 pairs.
                # 2415 pairs * cost of check.
                # Cost of check: compute needed points, enumerate candidates.
                # Let's try a random subset of pairs first, then all if time permits.
                pairs_to_try = []
                for i in range(n):
                    for j in range(i+1, n):
                        pairs_to_try.append((i, j))
                random.shuffle(pairs_to_try)
                
                for i, j in pairs_to_try:
                    if time.time() > deadline - 10:
                        break
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    # Compute triplets uniquely covered by b1 or b2
                    # unique_1 = triplets in b1 with count 1
                    # unique_2 = triplets in b2 with count 1
                    # But if a triplet is in both, and count is 2, it's not unique to either.
                    # We need to cover all triplets that will become uncovered if we remove b1 and b2.
                    # These are triplets where count == 1 AND (in b1 OR in b2).
                    
                    needed_triplets = []
                    # Get ids of b1 and b2
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                    
                    # A triplet is "needed" if it is in b1 or b2, and its count is 1.
                    # Wait, if count is 1, it must be in ONLY one of them (since we are removing both, and if it was in both, count would be >= 2).
                    # So we just check all triplets in b1 and b2.
                    
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        # Both are redundant (should have been caught by step 1)
                        # Remove both
                        # Careful with indices
                        # Remove j first (higher index)
                        b_remove = blocks.pop(j)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    # Find points in needed_triplets
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    # Enumerate blocks containing points_needed
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        # Check if new_block covers all needed_triplets
                        # And new_block is not b1 or b2
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        # Check coverage
                        covers = True
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        for tid in needed_triplets:
                            if tid not in new_ids:
                                covers = False
                                break
                                
                        if covers:
                            # Perform swap
                            # Remove b1, b2
                            # Order: pop j, then i
                            b_remove = blocks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                                
                            # Add new_block
                            blocks.append(new_block)
                            for triplet in itertools.combinations(new_block, t):
                                counts[triplet_to_id[triplet]] += 1
                                
                            improved = True
                            found_replacement = True
                            break
                            
                    if improved:
                        break
                        
            # 3. Try 1-for-1
            if not improved:
                # Pick random block
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = []
                for tid in ids_b1:
                    if counts[tid] == 1:
                        needed_triplets.append(tid)
                        
                if not needed_triplets:
                    # Redundant, handled by step 1
                    continue
                
                points_needed = set()
                for tid in needed_triplets:
                    trip = id_to_triplet[tid]
                    points_needed.update(trip)
                    
                if len(points_needed) > k:
                    continue
                    
                for new_block in blocks_containing(points_needed):
                    if set(new_block) == set(b1):
                        continue
                    new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                    covers = True
                    for tid in needed_triplets:
                        if tid not in new_ids:
                            covers = False
                            break
                    if covers:
                        # Replace
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        blocks.append(new_block)
                        for triplet in itertools.combinations(new_block, t):
                            counts[triplet_to_id[triplet]] += 1
                        improved = True
                        break
                        
            if not improved:
                # Kick: Randomly modify a block to escape local minimum?
                # Or just break and try a new seed?
                # For now, let's just break the inner loop and rely on multiple seeds
                break
                
            no_improve_count = 0 if improved else no_improve_count + 1
            
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times with different random seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Greedy init
        init_blocks = greedy_construct()
        
        # Optimize
        opt_blocks = optimize(init_blocks)
        
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
        best_blocks = greedy_construct()
        
    return best_blocks

if __name__ == "__main__":
    pass
