import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_blocks = list(itertools.combinations(points, k))
    # Precompute mask for each k-block
    block_masks = []
    for b in k_blocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            mask |= 1 << t_sub_to_idx[ts]
        block_masks.append(mask)
        
    n_blocks = len(k_blocks)
    
    def prune(blocks):
        # blocks is list of indices into k_blocks
        if not blocks: return []
        counts = [0] * n_t
        # Map block index to list of t-subset indices it covers?
        # We already have masks. It's faster to iterate bits?
        # Or precompute list of t-subset indices for each block?
        # 50k blocks * 35 subsets = 1.7M entries. Memory OK.
        # Let's precompute block_t_indices to speed up pruning.
        
        # Actually, let's just use the mask for pruning too?
        # Checking if (mask & ~others) == 0 is hard.
        # Counting is better.
        
        # Precompute t-indices for each block once globally?
        # Yes.
        
        # Let's assume block_t_indices is available.
        
        active = [True] * len(blocks)
        # Initialize counts
        for b_idx in blocks:
            for ts_idx in block_t_indices[b_idx]:
                counts[ts_idx] += 1
                
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(blocks):
                if not active[i]: continue
                # Check if any ts in this block has count 1
                is_essential = False
                for ts_idx in block_t_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    active[i] = False
                    changed = True
                    for ts_idx in block_t_indices[b_idx]:
                        counts[ts_idx] -= 1
        return [b_idx for i, b_idx in enumerate(blocks) if active[i]]

    # Precompute block_t_indices
    block_t_indices = []
    for b in k_blocks:
        indices = []
        for ts in itertools.combinations(b, t):
            indices.append(t_sub_to_idx[ts])
        block_t_indices.append(indices)

    def run_greedy(seed):
        random.seed(seed)
        uncovered = (1 << n_t) - 1
        selected_blocks = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            # Optimization: If we find a block covering ALL remaining, we can stop early?
            # Unlikely.
            
            for i in range(n_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                break # Should not happen if uncovered != 0
            
            # Random tie break
            chosen = random.choice(best_indices)
            selected_blocks.append(chosen)
            uncovered &= ~block_masks[chosen]
            
        # Prune
        pruned = prune(selected_blocks)
        return pruned

    # Main loop
    best_solution = None
    best_size = float('inf')
    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    seed = 0
    while time.time() - start_time < timeout:
        sol = run_greedy(seed)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            # If we hit the lower bound or a very good number, we might stop?
            # Lower bound is 131. Best known 151.
            if best_size <= 140: # Arbitrary good threshold
                break
        seed += 1
        
    # Convert indices to actual blocks
    return [list(k_blocks[i]) for i in best_solution]
