import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Iterated Local Search 
    with Greedy Initialization and Redundancy Removal.
    """
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Time limit 120s
    start_time = time.time()
    time_limit = 115 # Leave buffer
    
    max_restarts = 200 # Upper bound, will stop by time
    
    for seed in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        np.random.seed(seed)
        
        blocks = []
        # coverage[i] = number of blocks covering all_t_subsets[i]
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        
        # --- Phase 1: Greedy Construction ---
        # Maintain a set of uncovered indices for fast selection
        uncovered_indices = set(range(num_t_subsets))
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            start_idx = random.choice(tuple(uncovered_indices))
            start_subset = all_t_subsets[start_idx]
            
            block = list(start_subset)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # To speed up, we can estimate gain.
                # Exact gain: count how many t-subsets containing x and (t-1) from block are uncovered.
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Generate (t-1) subsets from current block
                    for rest in itertools.combinations(block, t - 1):
                        T = tuple(sorted(rest + (x,)))
                        idx = subset_to_idx.get(T)
                        if idx is not None and idx in uncovered_indices:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = subset_to_idx[ts]
                coverage[idx] += 1
                if coverage[idx] == 1:
                    uncovered_indices.discard(idx)
        
        # --- Phase 2: Redundancy Removal ---
        # We do this iteratively until no more blocks can be removed
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to make removal easier? Or random order.
            # Random order helps avoid bias.
            indices_to_remove = []
            for i in range(len(blocks)):
                block = blocks[i]
                # Check if all t-subsets of this block are covered by > 1 block
                # i.e., coverage[ts] > 1 for all ts in block
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    idx = subset_to_idx[ts]
                    if coverage[idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    indices_to_remove.append(i)
            
            if indices_to_remove:
                # Remove them
                for i in reversed(indices_to_remove):
                    block = blocks.pop(i)
                    for ts in itertools.combinations(block, t):
                        idx = subset_to_idx[ts]
                        coverage[idx] -= 1
                changed = True
        
        # --- Phase 3: Local Search (Swaps) ---
        # Try to improve by swapping blocks
        # This can be expensive, so limit steps
        local_steps = 100
        for _ in range(local_steps):
            if time.time() - start_time > time_limit:
                break
            
            if not blocks:
                break
                
            # Pick a random block to replace
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            
            # Generate a random new block
            new_block = sorted(random.sample(range(v), k))
            
            # Check if swap is valid
            # Optimization: Check if new_block is identical to old_block
            if old_block == new_block:
                continue
            
            # Proper Check:
            # A swap is valid if every t-subset in old_block that is NOT in new_block
            # has coverage > 1 (before decrement).
            
            new_block_set = set(itertools.combinations(new_block, t))
            
            valid = True
            for ts in itertools.combinations(old_block, t):
                if ts not in new_block_set:
                    idx = subset_to_idx[ts]
                    if coverage[idx] <= 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # If valid, perform the swap
            # Decrement old
            for ts in itertools.combinations(old_block, t):
                idx = subset_to_idx[ts]
                coverage[idx] -= 1
            
            # Increment new
            for ts in itertools.combinations(new_block, t):
                idx = subset_to_idx[ts]
                coverage[idx] += 1
            
            blocks[i] = new_block
            
            # After swap, run redundancy removal again
            changed = True
            while changed:
                changed = False
                indices_to_remove = []
                for j in range(len(blocks)):
                    block = blocks[j]
                    is_redundant = True
                    for ts in itertools.combinations(block, t):
                        idx = subset_to_idx[ts]
                        if coverage[idx] <= 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        indices_to_remove.append(j)
                
                if indices_to_remove:
                    for j in reversed(indices_to_remove):
                        block = blocks.pop(j)
                        for ts in itertools.combinations(block, t):
                            idx = subset_to_idx[ts]
                            coverage[idx] -= 1
                    changed = True
        
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
    return best_blocks
