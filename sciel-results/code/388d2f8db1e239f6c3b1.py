import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin for safety
    
    # Pre-computation
    all_points = list(range(v))
    t_subsets = list(itertools.combinations(all_points, t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    block_subsets = [[] for _ in range(n_blocks)]
    for b_idx, block in enumerate(k_blocks):
        for sub in itertools.combinations(block, t):
            block_subsets[b_idx].append(subset_to_idx[sub])
    
    # Build numpy matrix: coverage[block][subset] = 1 if block covers subset
    # This is n_blocks x n_subsets, could be large but manageable
    # n_blocks = C(19,8) = 75582, n_subsets = C(19,5) = 11628
    # That's ~878 million entries - too large for a dense matrix
    # Use sparse representation instead
    
    # For each subset, which blocks cover it?
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx in range(n_blocks):
        for s_idx in block_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)
    
    # Greedy construction with numpy acceleration
    def run_greedy(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        selected = []
        selected_arr = np.zeros(n_blocks, dtype=np.bool_)
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        
        while uncovered.any():
            if time.time() > deadline - 3:
                break
            
            # Find best block: maximize number of uncovered subsets covered
            # For each block, gain = sum of uncovered[s] for s in block_subsets[b]
            # Vectorize: for each block, compute dot product of its subsets with uncovered
            
            best_block = -1
            best_gain = 0
            
            # Chunk processing to avoid memory issues
            # Each block covers C(8,5) = 56 subsets
            # We can compute gains in batches
            
            batch_size = 5000
            for start in range(0, n_blocks, batch_size):
                end = min(start + batch_size, n_blocks)
                
                # Skip already selected
                batch_indices = np.arange(start, end)
                mask = ~selected_arr[batch_indices]
                if not mask.any():
                    continue
                batch_indices = batch_indices[mask]
                
                # For each block in batch, compute gain
                # block_subsets[b] is a list of subset indices
                # gain[b] = uncovered[block_subsets[b]].sum()
                
                gains = np.zeros(len(batch_indices), dtype=np.int32)
                for i, b_idx in enumerate(batch_indices):
                    subs = block_subsets[b_idx]
                    gains[i] = uncovered[subs].sum()
                
                # Find max
                local_max = np.argmax(gains)
                if gains[local_max] > best_gain:
                    best_gain = gains[local_max]
                    best_block = batch_indices[local_max]
            
            if best_block == -1 or best_gain == 0:
                break
            
            selected.append(best_block)
            selected_arr[best_block] = True
            
            # Update uncovered and cover_count
            subs = block_subsets[best_block]
            uncovered[subs] = False
            cover_count[subs] += 1
        
        return selected, cover_count
    
    # Redundancy removal
    def remove_redundant(selected, cover_count):
        if not selected:
            return selected, cover_count
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Try removing each block
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                b_idx = selected[i]
                subs = block_subsets[b_idx]
                # Check if all subsets in this block are covered by other blocks
                # i.e., cover_count[s] > 1 for all s in subs
                if np.all(cover_count[subs] > 1):
                    # Remove
                    cover_count[subs] -= 1
                    selected.pop(i)
                    changed = True
        
        return selected, cover_count
    
    # Run multiple greedy variants with different random seeds
    best_selected = None
    best_count = float('inf')
    
    # First run: deterministic greedy (seed=0)
    selected, cover_count = run_greedy(seed=0)
    selected, cover_count = remove_redundant(selected, cover_count)
    best_selected = selected[:]
    best_count = len(selected)
    
    # Try more variants if time permits
    n_restarts = 5
    for attempt in range(1, n_restarts + 1):
        if time.time() > deadline - 10:
            break
        seed = attempt * 12345 + 678
        selected, cover_count = run_greedy(seed=seed)
        selected, cover_count = remove_redundant(selected, cover_count)
        if len(selected) < best_count:
            best_selected = selected[:]
            best_count = len(selected)
    
    # Final: try more aggressive redundancy removal with multiple passes
    if time.time() < deadline - 5 and best_selected:
        # Rebuild cover_count
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in best_selected:
            for s_idx in block_subsets[b_idx]:
                cover_count[s_idx] += 1
        
        best_selected, cover_count = remove_redundant(best_selected, cover_count)
    
    # Convert to final blocks
    final_blocks = [list(k_blocks[i]) for i in best_selected]
    
    return final_blocks
