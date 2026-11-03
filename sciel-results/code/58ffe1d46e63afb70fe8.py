import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation
    points = range(v)
    all_t_subs = list(itertools.combinations(points, t))
    num_t_subs = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute which t-subsets each k-block covers
    block_t_indices = []
    for block in all_k_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_sub_to_idx[sub])
        block_t_indices.append(indices)
        
    # Precompute which k-blocks cover each t-subset
    # blocks_containing[t_idx] = list of k_block_indices
    blocks_containing = [[] for _ in range(num_t_subs)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            blocks_containing[t_idx].append(b_idx)
            
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        random.seed() # or use a counter
        uncovered = set(range(num_t_subs))
        chosen_blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            s = random.choice(list(uncovered)) # list() conversion is O(N), maybe use pop?
            # Actually, set.pop() is O(1) but removes it. We want to keep it if we don't pick the best block?
            # No, we pick a block that covers it. It will be removed anyway.
            # But we need to pick one to start the candidate search.
            # Using list(uncovered) is expensive if uncovered is large.
            # Alternative: iterate through uncovered and pick one?
            # Or just pick the first one? Randomness is better.
            # Let's use a list for uncovered? No, set is better for removal.
            # To pick random element from set efficiently:
            # In Python 3.7+, list(set) is O(N).
            # If N is 11628, it's fine.
            
            candidates = blocks_containing[s]
            best_b_idx = -1
            best_gain = 0
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            chosen_blocks.append(best_b_idx)
            # Remove covered subsets
            for t_idx in block_t_indices[best_b_idx]:
                uncovered.discard(t_idx)
                
        # Post-processing: Iterative Deletion
        cover_count = [0] * num_t_subs
        for b_idx in chosen_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(chosen_blocks)):
                b_idx = chosen_blocks[i]
                all_covered = True
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        all_covered = False
                        break
                if not all_covered:
                    # Restore
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] += 1
                else:
                    # Remove
                    chosen_blocks.pop(i)
                    changed = True
                    break # Restart loop
                    
        if len(chosen_blocks) < best_count:
            best_count = len(chosen_blocks)
            best_blocks = [list(all_k_blocks[b_idx]) for b_idx in chosen_blocks]
            
        # If we found a very good solution, we might stop early?
        # Lower bound is 264. If we hit 265, we are close.
        # But let's keep searching for a bit more.
        
    if best_blocks is None:
        # Fallback: should not happen
        return []
        
    return best_blocks
