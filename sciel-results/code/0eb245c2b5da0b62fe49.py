import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute which t-subsets are covered by each k-block
    block_t_indices = []
    for block in all_k_blocks:
        t_subs_in_block = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(np.array(t_subs_in_block, dtype=np.int32))
    
    best_blocks = None
    best_size = float('inf')
    
    num_trials = 0
    
    while time.time() < deadline - 5.0: # Leave 5s for final cleanup
        num_trials += 1
        rng = np.random.default_rng(num_trials)
        
        # State: uncovered count for each t-subset
        uncovered_count = np.ones(n_t_subsets, dtype=np.int32)
        selected_block_indices = []
        
        # Randomized Greedy
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Calculate gain for each block
            gains = np.zeros(n_k_blocks, dtype=np.int32)
            for i, t_indices in enumerate(block_t_indices):
                gains[i] = np.sum(uncovered_count[t_indices])
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break # All covered
                
            # Find all blocks with max gain
            best_indices = np.where(gains == max_gain)[0]
            
            # Randomly pick one from the best candidates
            if len(best_indices) > 1:
                chosen_idx = best_indices[rng.integers(0, len(best_indices))]
            else:
                chosen_idx = best_indices[0]
                
            selected_block_indices.append(chosen_idx)
            
            # Update uncovered counts
            t_indices = block_t_indices[chosen_idx]
            uncovered_count[t_indices] -= 1
            
        # If we broke out due to time, check if covering is complete
        if np.any(uncovered_count > 0):
            continue
            
        # Redundancy Removal
        n_selected = len(selected_block_indices)
        active = [True] * n_selected
        
        # Initial cover count
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for idx in selected_block_indices:
            t_indices = block_t_indices[idx]
            cover_count[t_indices] += 1
            
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(n_selected):
                if not active[i]:
                    continue
                t_indices = block_t_indices[selected_block_indices[i]]
                # If no t-subset in this block has count 1, it's redundant
                if not np.any(cover_count[t_indices] == 1):
                    active[i] = False
                    cover_count[t_indices] -= 1
                    changed = True
                    
        # Construct final list for this trial
        current_blocks = [all_k_blocks[selected_block_indices[i]] for i in range(n_selected) if active[i]]
        
        if len(current_blocks) < best_size:
            best_size = len(current_blocks)
            best_blocks = current_blocks
            
    if best_blocks is None:
        # Fallback: Deterministic greedy
        uncovered_count = np.ones(n_t_subsets, dtype=np.int32)
        selected_block_indices = []
        while True:
            gains = np.zeros(n_k_blocks, dtype=np.int32)
            for i, t_indices in enumerate(block_t_indices):
                gains[i] = np.sum(uncovered_count[t_indices])
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_indices = np.where(gains == max_gain)[0]
            chosen_idx = best_indices[0]
            selected_block_indices.append(chosen_idx)
            t_indices = block_t_indices[chosen_idx]
            uncovered_count[t_indices] -= 1
            
        # Redundancy removal
        n_selected = len(selected_block_indices)
        active = [True] * n_selected
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for idx in selected_block_indices:
            t_indices = block_t_indices[idx]
            cover_count[t_indices] += 1
        changed = True
        while changed:
            changed = False
            for i in range(n_selected):
                if not active[i]:
                    continue
                t_indices = block_t_indices[selected_block_indices[i]]
                if not np.any(cover_count[t_indices] == 1):
                    active[i] = False
                    cover_count[t_indices] -= 1
                    changed = True
        best_blocks = [all_k_blocks[selected_block_indices[i]] for i in range(n_selected) if active[i]]
        
    return best_blocks
