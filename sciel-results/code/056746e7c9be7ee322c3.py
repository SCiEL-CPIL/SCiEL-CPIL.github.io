import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if block i covers t-subset j
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # Helper: Greedy construction given current uncovered mask and allowed blocks
    def greedy_construct(uncovered_mask, allowed_blocks_mask, noise_scale=0.0):
        selected = []
        unc = uncovered_mask.copy()
        
        while np.any(unc):
            # Calculate gains: how many currently uncovered t-subsets does each block cover?
            # gains[i] = sum(Cover[i, j] for j where unc[j]==1)
            gains = Cover[allowed_blocks_mask] @ unc
            
            if np.max(gains) == 0:
                break # Cannot cover more
            
            # Add noise to explore different solutions
            if noise_scale > 0:
                # Only add noise to blocks that have positive gain to avoid picking useless blocks
                noise = np.random.rand(n_blocks) * noise_scale
                noise[~allowed_blocks_mask] = 0
                gains += noise
            
            best_idx_in_subset = np.argmax(gains)
            # Map back to original index
            # Since allowed_blocks_mask is boolean, we need to find the original index
            # It's faster to keep a list of allowed indices
            # For now, let's assume allowed_blocks_mask is always all True in the main loop, 
            # or we pass a list of indices.
            # Let's refactor to pass list of indices for flexibility.
            
            # This inner function is getting complex with masks. 
            # Let's use a list of candidate indices.
            pass
        
        # The above greedy_construct is tricky with masks. 
        # Let's write a cleaner version using lists of indices.
        pass

    # Better approach: 
    # 1. Initial Greedy
    # 2. Reduction
    # 3. Iterated Local Search (Destroy/Repair)
    
    def get_gains(unc, block_indices):
        # unc: 1D array bool/uint8
        # block_indices: list of int
        if not block_indices:
            return np.array([])
        # Extract rows for these blocks
        # This is slow if done every step. 
        # Better to maintain a matrix of active blocks?
        # For v=19, k=7, n_blocks = 50388. n_tsubs = 3876.
        # Matrix is 50388 x 3876 ~ 195M elements. 
        # Cover @ unc is a matrix-vector product. 
        # 50388 * 3876 operations. 
        # NumPy is fast, ~1ms per iteration? 
        # Greedy needs ~150 iterations. 150ms. 
        # So we can do many restarts.
        
        pass

    # Let's implement a standard fast greedy first.
    def run_greedy(noise=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        selected_indices = []
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        
        while np.sum(uncovered) > 0:
            gains = Cover @ uncovered
            if noise > 0:
                # Add small random noise to break ties and explore
                noise_arr = np.random.rand(n_blocks) * noise
                gains = gains + noise_arr
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            selected_indices.append(best_idx)
            uncovered = uncovered & (Cover[best_idx] == 0)
            
        return selected_indices

    def reduce_cover(selected_indices):
        # Remove redundant blocks
        if not selected_indices:
            return selected_indices
            
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        changed = True
        while changed:
            changed = False
            new_selected = []
            for idx in selected_indices:
                # Block is redundant if it doesn't cover any t-subset uniquely
                # i.e., for all j where Cover[idx,j]==1, cover_counts[j] > 1
                # Equivalently: not (Cover[idx] == 1 AND cover_counts == 1).any()
                is_redundant = not np.any((Cover[idx] == 1) & (cover_counts == 1))
                if is_redundant:
                    cover_counts -= Cover[idx]
                    changed = True
                else:
                    new_selected.append(idx)
            selected_indices = new_selected
            
        return selected_indices

    def destroy_and_repair(current_cover, num_to_destroy, noise=0.1):
        # Remove 'num_to_destroy' random blocks
        if len(current_cover) <= num_to_destroy:
            return current_cover
            
        current_cover = list(current_cover)
        to_remove = random.sample(range(len(current_cover)), num_to_destroy)
        
        # Identify which t-subsets become uncovered
        # We need to recalculate coverage or just mark them.
        # Let's just rebuild the uncovered mask from scratch for simplicity, 
        # as n_tsubs is small.
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = [idx for i, idx in enumerate(current_cover) if i not in set(to_remove)]
        
        for idx in remaining:
            uncovered = uncovered & (Cover[idx] == 0)
            
        # Now greedily cover the remaining gaps
        # We can use all blocks, or just those not in remaining? 
        # Usually better to allow all, but prefer those not in remaining to avoid cycles?
        # Let's allow all blocks.
        
        # We want to add blocks to 'remaining' to cover 'uncovered'.
        # We run a greedy loop, but we already have 'remaining'.
        # We just need to add new blocks.
        
        new_blocks = []
        unc = uncovered.copy()
        
        while np.sum(unc) > 0:
            gains = Cover @ unc
            if noise > 0:
                noise_arr = np.random.rand(n_blocks) * noise
                gains = gains + noise_arr
                
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
                
            new_blocks.append(best_idx)
            unc = unc & (Cover[best_idx] == 0)
            
        candidate_cover = remaining + new_blocks
        candidate_cover = reduce_cover(candidate_cover)
        
        return candidate_cover

    best_cover = []
    best_count = float('inf')
    
    # Phase 1: Initial Greedy with different noise levels
    for noise in [0.0, 0.01, 0.05, 0.1]:
        if time.time() > deadline - 30:
            break
        cover = run_greedy(noise=noise)
        cover = reduce_cover(cover)
        if len(cover) < best_count:
            best_count = len(cover)
            best_cover = cover
            
    # Phase 2: Iterated Local Search (Destroy and Repair)
    # Start from best_cover
    current_cover = list(best_cover)
    
    # Adaptive destruction size
    base_destroy = 5
    max_destroy = 20
    
    while time.time() < deadline - 10:
        # Randomize destruction size
        destroy_size = random.randint(1, base_destroy)
        
        # Add some noise to the repair phase
        noise = random.uniform(0.0, 0.05)
        
        new_cover = destroy_and_repair(current_cover, destroy_size, noise)
        
        if len(new_cover) < len(current_cover):
            current_cover = new_cover
            if len(current_cover) < best_count:
                best_count = len(current_cover)
                best_cover = current_cover
                # If we found a new best, maybe increase exploration?
                # Or just continue.
        else:
            # With some probability, accept worse solutions to escape local minima?
            # For covering designs, it's hard to define "temperature".
            # Let's just stick to hill climbing for now, but with random restarts if stuck.
            # If no improvement in last 50 iterations, do a full restart?
            pass
        
        # If we are stuck for a while, try a larger destruction
        # Simple heuristic: if len(new_cover) == len(current_cover), 
        # the swap didn't help. 
        # Let's just keep going.
        
    # Final reduction
    best_cover = reduce_cover(best_cover)
    
    # Convert indices to blocks
    blocks = [all_kblocks[i] for i in best_cover]
    
    return blocks
