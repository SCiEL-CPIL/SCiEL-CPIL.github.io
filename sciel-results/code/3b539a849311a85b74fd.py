import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145 # 5s safety margin
    
    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip
        
    # Precompute all blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Helper to get mask of a block
    def get_block_mask(block):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        return mask

    # Precompute masks for all blocks
    # This might take a few seconds
    block_masks = [get_block_mask(b) for b in all_blocks]
    
    # Helper to enumerate blocks containing a set of points (for local search)
    def blocks_containing(points):
        points = set(points)
        if len(points) > k:
            return []
        remaining = [p for p in range(v) if p not in points]
        need = k - len(points)
        if need < 0:
            return []
        for comb in itertools.combinations(remaining, need):
            yield tuple(sorted(points.union(comb)))

    # Global Greedy Construction
    def greedy_global(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            # Iterate through all precomputed blocks
            for i in range(n_blocks):
                mask = block_masks[i]
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if not best_indices:
                break # Should not happen if valid covering exists
                
            # Pick randomly among best to allow diversity
            idx = random.choice(best_indices)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
            
        return [all_blocks[i] for i in selected_indices]

    # Local Greedy Construction (Faster, restricted)
    def greedy_local(seed):
        random.seed(seed)
        uncovered = (1 << n_triplets) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triplet
            # Find a set bit
            lsb = uncovered & -uncovered
            idx = lsb.bit_length() - 1
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet
            best_block = None
            best_gain = -1
            
            for block in blocks_containing(start_triplet):
                b_mask = get_block_mask(block)
                gain = (b_mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_mask = b_mask
                    
            if best_block:
                blocks.append(best_block)
                uncovered &= ~best_mask
            else:
                # Fallback: shouldn't happen
                break
                
        return blocks

    # Local Search Optimizer
    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        
        def recalc_counts(b_list):
            nonlocal counts
            counts[:] = 0
            for b in b_list:
                for triplet in itertools.combinations(b, t):
                    counts[triplet_to_id[triplet]] += 1
        
        recalc_counts(blocks)
        
        max_iters = 5000
        no_improve_count = 0
        
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
                        blocks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                        
            # 2. Try 2-for-1
            if not improved:
                pairs_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs_to_try)
                
                for i, j in pairs_to_try:
                    if time.time() > deadline - 10:
                        break
                        
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                    ids_b2 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b2, t))
                    
                    needed_triplets = []
                    all_involved = ids_b1 | ids_b2
                    for tid in all_involved:
                        if counts[tid] == 1:
                            needed_triplets.append(tid)
                            
                    if not needed_triplets:
                        # Both redundant
                        b_remove = blocks.pop(j)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        b_remove = blocks.pop(i)
                        for triplet in itertools.combinations(b_remove, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) > k:
                        continue
                        
                    found_replacement = False
                    for new_block in blocks_containing(points_needed):
                        if set(new_block) == set(b1) or set(new_block) == set(b2):
                            continue
                            
                        new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                        covers = all(tid in new_ids for tid in needed_triplets)
                        
                        if covers:
                            # Swap
                            b_remove = blocks.pop(j)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                            b_remove = blocks.pop(i)
                            for triplet in itertools.combinations(b_remove, t):
                                counts[triplet_to_id[triplet]] -= 1
                                
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
                i = random.randint(0, n-1)
                b1 = blocks[i]
                ids_b1 = set(triplet_to_id[triplet] for triplet in itertools.combinations(b1, t))
                
                needed_triplets = [tid for tid in ids_b1 if counts[tid] == 1]
                
                if needed_triplets:
                    points_needed = set()
                    for tid in needed_triplets:
                        trip = id_to_triplet[tid]
                        points_needed.update(trip)
                        
                    if len(points_needed) <= k:
                        for new_block in blocks_containing(points_needed):
                            if set(new_block) == set(b1):
                                continue
                            new_ids = set(triplet_to_id[triplet] for triplet in itertools.combinations(new_block, t))
                            if all(tid in new_ids for tid in needed_triplets):
                                b_remove = blocks.pop(i)
                                for triplet in itertools.combinations(b_remove, t):
                                    counts[triplet_to_id[triplet]] -= 1
                                blocks.append(new_block)
                                for triplet in itertools.combinations(new_block, t):
                                    counts[triplet_to_id[triplet]] += 1
                                improved = True
                                break
                                
            if not improved:
                break
                
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Strategy: Mix of Global and Local Greedy
    # Global Greedy is slower but potentially better initialization
    # Local Greedy is faster, allows more runs
    
    candidates = []
    
    # Run Local Greedy 10 times
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        blocks = greedy_local(seed)
        candidates.append(blocks)
        
    # Run Global Greedy 5 times
    for seed in range(10, 15):
        if time.time() > deadline - 20:
            break
        blocks = greedy_global(seed)
        candidates.append(blocks)
        
    # Optimize all candidates
    for blocks in candidates:
        if time.time() > deadline - 10:
            break
            
        opt_blocks = optimize(blocks)
        
        # Verify
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
        best_blocks = greedy_local(0)
        
    return best_blocks

if __name__ == "__main__":
    pass
