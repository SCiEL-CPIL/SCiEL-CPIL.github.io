import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin for safety
    
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Precompute masks for all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << subset_to_idx[sub]
        block_masks[b] = mask
    
    # Precompute for each point x, the list of t-subsets containing x
    # This helps in gain calculation
    point_subsets = [[] for _ in range(v)]
    for sub in t_subsets:
        for x in sub:
            point_subsets[x].append(subset_to_idx[sub])
    
    best_blocks = None
    best_size = float('inf')
    
    # Strategy 0: Standard greedy with lowest bit
    # Strategy 1: Randomized greedy - pick random uncovered subset
    # Strategy 2: Greedy with tie-breaking randomization
    # Strategy 3: Greedy with "hard subset" preference (pick uncovered subset with fewest covering blocks available)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        strategy = attempt % 4
        
        uncovered = (1 << n_subsets) - 1
        blocks = []
        counts = [0] * n_subsets
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick which uncovered subset to start with
            if strategy == 0:
                # Lowest set bit
                idx = (uncovered & -uncovered).bit_length() - 1
            elif strategy == 1:
                # Random set bit
                pop = uncovered.bit_count()
                if pop == 0:
                    break
                rank = random.randrange(pop)
                # Find rank-th set bit
                temp = uncovered
                found = -1
                for i in range(n_subsets):
                    if temp & (1 << i):
                        rank -= 1
                        if rank < 0:
                            found = i
                            break
                idx = found
                if idx == -1:
                    idx = (uncovered & -uncovered).bit_length() - 1
            elif strategy == 2:
                # Pick uncovered subset that has fewest potential covering blocks
                # Approximate: pick one where the points are "rare"
                # For speed, just use lowest bit but with slight randomization
                pop = uncovered.bit_count()
                if pop <= 10:
                    # Small: pick random
                    bits = []
                    temp = uncovered
                    for i in range(n_subsets):
                        if temp & (1 << i):
                            bits.append(i)
                    idx = random.choice(bits)
                else:
                    idx = (uncovered & -uncovered).bit_length() - 1
            else:
                # Strategy 3: Weighted random - prefer subsets whose points appear in fewer remaining blocks
                # For simplicity, just use random among a sample
                sample_size = min(50, uncovered.bit_count())
                candidates = []
                temp = uncovered
                i = 0
                while len(candidates) < sample_size and i < n_subsets:
                    if temp & (1 << i):
                        candidates.append(i)
                    i += 1
                if candidates:
                    idx = random.choice(candidates)
                else:
                    idx = (uncovered & -uncovered).bit_length() - 1
            
            first_subset = t_subsets[idx]
            block = list(first_subset)
            
            # Add k-t more points greedily
            while len(block) < k:
                best_gain = -1
                best_point = -1
                best_mask = 0
                
                for x in range(v):
                    if x in block:
                        continue
                    temp_block = tuple(sorted(block + [x]))
                    
                    # Compute mask of t-subsets in temp_block that are currently uncovered
                    # Only need to check t-subsets of temp_block
                    mask_x = 0
                    for sub in itertools.combinations(temp_block, t):
                        mask_x |= 1 << subset_to_idx[sub]
                    
                    gain = (mask_x & uncovered).bit_count()
                    if gain > best_gain or (gain == best_gain and best_point != -1 and x < best_point and random.random() < 0.1):
                        best_gain = gain
                        best_point = x
                        best_mask = mask_x
                        
                if best_point == -1:
                    break
                block.append(best_point)
                block.sort()
                
            # Finalize block
            final_block = tuple(block)
            blocks.append(final_block)
            
            # Update uncovered and counts
            mask = block_masks[final_block]
            uncovered &= ~mask
            for sub in itertools.combinations(final_block, t):
                counts[subset_to_idx[sub]] += 1
        
        if time.time() > deadline:
            break
            
        # Pruning Phase - more efficient: remove all redundants in each pass
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(blocks)):
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if counts[subset_to_idx[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            # Remove from end to preserve indices
            for i in reversed(to_remove):
                b = blocks[i]
                blocks.pop(i)
                for sub in itertools.combinations(b, t):
                    counts[subset_to_idx[sub]] -= 1
                changed = True
            if to_remove:
                changed = True
            else:
                changed = False
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
