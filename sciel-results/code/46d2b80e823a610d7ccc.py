import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # Precompute all k-blocks and their t-subsets
    k_blocks = list(itertools.combinations(points, k))
    block_t_subsets = []
    t_subset_to_blocks = {ts: [] for ts in t_subsets}
    
    for b_idx, block in enumerate(k_blocks):
        # Get indices of t-subsets contained in this block
        b_t_subsets = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            b_t_subsets.append(idx)
            t_subset_to_blocks[ts].append(b_idx)
        block_t_subsets.append(b_t_subsets)
        
    # State: coverage count for each t-subset
    covered = [0] * n_t_subsets
    blocks_used = []
    
    # Find uncovered t-subsets efficiently? 
    # We can just iterate through all and pick one. 
    # Or maintain a set of uncovered indices.
    uncovered_indices = set(range(n_t_subsets))
    
    while uncovered_indices:
        if time.time() > deadline:
            break
            
        # Pick an uncovered t-subset to target
        # Heuristic: Pick the one that is covered by the fewest potential blocks?
        # All uncovered ones are covered by exactly C(v-t, k-t) blocks.
        # So any uncovered one is fine. Let's pick the first one for determinism.
        target_idx = next(iter(uncovered_indices))
        target_ts = t_subsets[target_idx]
        
        # Get candidate blocks containing this target t-subset
        candidate_block_indices = t_subset_to_blocks[target_ts]
        
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in candidate_block_indices:
            # Calculate gain: number of uncovered t-subsets in this block
            gain = 0
            for ts_idx in block_t_subsets[b_idx]:
                if ts_idx in uncovered_indices:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            # Should not happen if there is an uncovered t-subset
            break
            
        # Add the best block
        blocks_used.append(list(k_blocks[best_block_idx]))
        
        # Update coverage
        for ts_idx in block_t_subsets[best_block_idx]:
            covered[ts_idx] += 1
            if covered[ts_idx] == 1:
                uncovered_indices.discard(ts_idx)
                
    return blocks_used
