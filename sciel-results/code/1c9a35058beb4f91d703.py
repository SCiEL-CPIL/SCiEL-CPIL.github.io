import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    
    # 2. Precompute bitmasks for t-subsets (not strictly needed for logic, but good for consistency)
    # We actually don't need t_masks if we use indices directly.
    
    # 3. For each t-subset, find all k-blocks containing it
    t_to_blocks = [[] for _ in range(n_t)]
    for i, t_subset in enumerate(t_subsets):
        t_set = set(t_subset)
        available = [p for p in range(v) if p not in t_set]
        for add_pts in itertools.combinations(available, k - t):
            full_block = tuple(sorted(t_subset + add_pts))
            mask = 0
            for p in full_block:
                mask |= (1 << p)
            t_to_blocks[i].append(mask)
            
    # 4. Build reverse map: block_mask -> list of t-subset indices
    block_to_t_indices = {}
    for i in range(n_t):
        for mask in t_to_blocks[i]:
            if mask not in block_to_t_indices:
                block_to_t_indices[mask] = []
            block_to_t_indices[mask].append(i)
            
    # 5. Greedy Trials
    best_blocks = None
    best_len = float('inf')
    
    trial = 0
    while time.time() < deadline:
        trial += 1
        random.seed(trial)
        
        # Optimized uncovered management
        is_uncovered = [True] * n_t
        uncovered_list = list(range(n_t))
        uncovered_count = n_t
        blocks = []
        
        while uncovered_count > 0:
            # Pick random valid index from uncovered_list
            # Find a valid index
            # Swap-remove strategy:
            # Pick random pos
            pos = random.randint(0, len(uncovered_list) - 1)
            idx = uncovered_list[pos]
            
            # If it's still uncovered, use it
            # If not, swap with last, pop, and retry? 
            # Actually, if we maintain the invariant that uncovered_list only contains uncovered items,
            # we must remove items when they become covered.
            
            # Let's stick to the simple method:
            # Pick random idx from uncovered_list.
            # If is_uncovered[idx] is False, remove it from list (swap-pop) and try again.
            # This is amortized O(1).
            
            while True:
                pos = random.randint(0, len(uncovered_list) - 1)
                idx = uncovered_list[pos]
                if is_uncovered[idx]:
                    break
                # Remove stale entry
                last_idx = uncovered_list[-1]
                uncovered_list[pos] = last_idx
                uncovered_list.pop()
                
            random_t_idx = idx
            candidates = t_to_blocks[random_t_idx]
            
            best_block_mask = None
            max_covered = -1
            
            for mask in candidates:
                covered_count = 0
                for t_idx in block_to_t_indices[mask]:
                    if is_uncovered[t_idx]:
                        covered_count += 1
                        if covered_count > max_covered:
                            max_covered = covered_count
                            
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_block_mask = mask
            
            if best_block_mask is None:
                break
                
            blocks.append(best_block_mask)
            
            # Mark covered
            for t_idx in block_to_t_indices[best_block_mask]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    uncovered_count -= 1
                    
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    # 6. Convert masks to lists
    if best_blocks is None:
        return []
        
    result = []
    for mask in best_blocks:
        pts = []
        for p in range(v):
            if mask & (1 << p):
                pts.append(p)
        result.append(pts)
        
    return result
