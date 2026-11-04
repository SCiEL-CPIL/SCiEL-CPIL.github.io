import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    # Precompute all possible blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map triplet to list of block indices containing it
    triplet_to_blocks = defaultdict(list)
    block_masks = [0] * n_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
            triplet_to_blocks[tid].append(i)
        block_masks[i] = mask

    def greedy_construct():
        uncovered = (1 << n_triplets) - 1
        blocks = []
        selected_indices = set()
        
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(n_blocks):
                if i in selected_indices:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
            
            blocks.append(all_blocks[best_idx])
            selected_indices.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return blocks

    def optimize(blocks):
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
        
        def get_unique_triplets(block, counts):
            uniques = []
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if counts[tid] == 1:
                    uniques.append(tid)
            return uniques

        max_iters = 2000
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
                        blocks.pop(i)
                        for triplet in itertools.combinations(b, t):
                            counts[triplet_to_id[triplet]] -= 1
                        improved = True
                        break

            # 2. Try 1-for-1
            if not improved:
                i = random.randint(0, n-1)
                b1 = blocks[i]
                unique_tids = get_unique_triplets(b1, counts)
                
                if not unique_tids:
                    continue
                
                # Find intersection of blocks covering all unique_tids
                # Start with list of blocks for first tid
                candidates = set(triplet_to_blocks[unique_tids[0]])
                for tid in unique_tids[1:]:
                    candidates.intersection_update(triplet_to_blocks[tid])
                    if not candidates:
                        break
                
                # Remove b1 from candidates
                b1_idx = block_to_idx.get(b1) # Need this map
                # We didn't create block_to_idx in this version. Let's add it.
                # Actually, we can just check if the block tuple matches.
                # But we have indices.
                # Let's assume we have block_to_idx.
                
                found = False
                for idx in candidates:
                    if all_blocks[idx] == b1:
                        continue
                    # Check if this block is a valid replacement
                    # It covers all unique_tids.
                    # We replace b1 with all_blocks[idx]
                    blocks.pop(i)
                    for triplet in itertools.combinations(b1, t):
                        counts[triplet_to_id[triplet]] -= 1
                    
                    new_block = all_blocks[idx]
                    blocks.append(new_block)
                    for triplet in itertools.combinations(new_block, t):
                        counts[triplet_to_id[triplet]] += 1
                    
                    improved = True
                    found = True
                    break
                
                if found:
                    continue

            # 3. Try 2-for-1
            if not improved:
                if n < 2:
                    break
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                
                b1 = blocks[i]
                b2 = blocks[j]
                
                needed_tids = set()
                for triplet in itertools.combinations(b1, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        needed_tids.add(tid)
                for triplet in itertools.combinations(b2, t):
                    tid = triplet_to_id[triplet]
                    if counts[tid] == 1:
                        needed_tids.add(tid)
                        
                if not needed_tids:
                    # Remove both
                    b_rem = blocks.pop(j)
                    for triplet in itertools.combinations(b_rem, t):
                        counts[triplet_to_id[triplet]] -= 1
                    b_rem = blocks.pop(i)
                    for triplet in itertools.combinations(b_rem, t):
                        counts[triplet_to_id[triplet]] -= 1
                    improved = True
                    continue
                
                # Find intersection
                candidates = set(triplet_to_blocks[list(needed_tids)[0]])
                for tid in list(needed_tids)[1:]:
                    candidates.intersection_update(triplet_to_blocks[tid])
                    if not candidates:
                        break
                
                found = False
                for idx in candidates:
                    if all_blocks[idx] == b1 or all_blocks[idx] == b2:
                        continue
                    
                    # Replace
                    b_rem = blocks.pop(j)
                    for triplet in itertools.combinations(b_rem, t):
                        counts[triplet_to_id[triplet]] -= 1
                    b_rem = blocks.pop(i)
                    for triplet in itertools.combinations(b_rem, t):
                        counts[triplet_to_id[triplet]] -= 1
                    
                    new_block = all_blocks[idx]
                    blocks.append(new_block)
                    for triplet in itertools.combinations(new_block, t):
                        counts[triplet_to_id[triplet]] += 1
                    
                    improved = True
                    found = True
                    break
                
                if found:
                    continue

            if not improved:
                no_improve += 1
                if no_improve > 20:
                    break
            else:
                no_improve = 0

        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Create block_to_idx map for 1-for-1 check
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        init_blocks = greedy_construct()
        opt_blocks = optimize(init_blocks)
        
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
                
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks
                
    if best_blocks is None:
        best_blocks = greedy_construct()
        
    return best_blocks

if __name__ == "__main__":
    pass
