import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Incidence data structures
    # block_cover_indices[b] = array of t-subset indices covered by block b
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
        
    # tsub_to_blocks[s] = list of block indices covering t-subset s
    # This is needed for the swap heuristic
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, inds in enumerate(block_cover_indices):
        for s_idx in inds:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # Convert tsub_to_blocks to numpy arrays for faster intersection logic if needed, 
    # but lists are fine for small k,t.
    # Actually, for swap, we need to find blocks that cover ALL subsets in a small set U.
    # If U = [s1, s2], we need b in tsub_to_blocks[s1] AND b in tsub_to_blocks[s2].
    
    # Dense matrix for fast greedy
    M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    for b_idx, inds in enumerate(block_cover_indices):
        M[b_idx, inds] = 1
        
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # We will run a few iterations with different random seeds/tie-breaks
    # Since greedy is fast, we can try a few times if time permits.
    # But local search is the main differentiator.
    
    for restart in range(3): # Max 3 restarts
        if time.time() > deadline - 20: # Leave 20s for final local search
            break
            
        # --- Greedy Phase ---
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        
        # Add slight noise to gains to break symmetry and explore different paths
        # We will re-calculate gains each step, so we can add noise to the selection
        while np.any(uncovered_mask):
            if time.time() > deadline - 5:
                break # Emergency stop, result might be invalid if uncovered remains
                
            gains = M @ uncovered_mask
            
            # Add small random noise to break ties
            # Scale noise so it doesn't override large gain differences, but breaks exact ties
            noise = rng.random(n_k_blocks) * 0.01
            gains_with_noise = gains + noise
            
            best_block = np.argmax(gains_with_noise)
            
            if gains[best_block] == 0:
                # Should not happen if cover is possible
                break
                
            selected_indices.append(best_block)
            
            # Update uncovered mask
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
            
        # If greedy didn't finish, skip this restart
        if np.any(uncovered_mask):
            continue
            
        # --- Local Search Phase ---
        # Initialize cover counts
        cover_count = np.zeros(n_t_subsets, dtype=np.int16)
        for b_idx in selected_indices:
            cover_count[block_cover_indices[b_idx]] += 1
            
        improved = True
        local_search_time_limit = deadline - 2 # 2s safety buffer
        
        while improved and time.time() < local_search_time_limit:
            improved = False
            
            # Step 1: Redundancy Removal
            # A block is redundant if all its covered t-subsets have count > 1
            redundant_mask = np.zeros(len(selected_indices), dtype=bool)
            
            # Vectorized-ish check. 
            # We can't easily vectorize the min over variable length arrays in a single op without padding, 
            # but len(selected_indices) is small (~150).
            for i, b_idx in enumerate(selected_indices):
                subs = block_cover_indices[b_idx]
                if np.min(cover_count[subs]) > 1:
                    redundant_mask[i] = True
                    
            if np.any(redundant_mask):
                # Remove redundant blocks
                # Update cover_count
                for i, b_idx in enumerate(selected_indices):
                    if redundant_mask[i]:
                        cover_count[block_cover_indices[b_idx]] -= 1
                        improved = True
                        
                # Filter list
                selected_indices = [b_idx for i, b_idx in enumerate(selected_indices) if not redundant_mask[i]]
                continue # Restart loop to check for new redundancies or swaps
            
            # Step 2: 1-Swap Search
            # Try to replace a block b_out with b_in
            # b_in must cover all t-subsets that are uniquely covered by b_out
            
            # Identify unique subsets for each block in current solution
            # unique_subs[b_idx] = list of s such that s in b_idx and cover_count[s] == 1
            
            # To speed up, only consider blocks with few unique subsets.
            # If a block has many unique subsets, it's hard to replace.
            
            candidates_out = []
            unique_subs_map = {}
            
            for b_idx in selected_indices:
                subs = block_cover_indices[b_idx]
                # Find subsets with count 1
                mask = cover_count[subs] == 1
                if np.any(mask):
                    u_subs = subs[mask]
                    # Limit to small sets for efficiency. 
                    # If > 2 unique subsets, intersection of block lists might be empty or rare.
                    if len(u_subs) <= 2: 
                        candidates_out.append(b_idx)
                        unique_subs_map[b_idx] = u_subs
            
            if not candidates_out:
                break # No simple swaps possible
            
            # Shuffle to avoid bias
            rng.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                if time.time() > local_search_time_limit:
                    break
                    
                u_subs = unique_subs_map[b_out]
                
                # Find candidate blocks b_in that cover all subsets in u_subs
                # Start with the list of blocks covering the first unique subset
                first_s = u_subs[0]
                potential_b_in = tsub_to_blocks[first_s]
                
                # Filter potential_b_in
                valid_b_in = []
                for b_in in potential_b_in:
                    if b_in in selected_indices:
                        continue
                    # Check if b_in covers all other subsets in u_subs
                    # Since len(u_subs) <= 2, this is fast
                    covers_all = True
                    for s in u_subs[1:]:
                        # Check if b_in covers s
                        # We can use M[b_in, s]
                        if M[b_in, s] == 0:
                            covers_all = False
                            break
                    if covers_all:
                        valid_b_in.append(b_in)
                        
                if valid_b_in:
                    # Pick one randomly to explore
                    b_in = valid_b_in[rng.integers(len(valid_b_in))]
                    
                    # Perform swap
                    # Remove b_out
                    selected_indices.remove(b_out)
                    cover_count[block_cover_indices[b_out]] -= 1
                    
                    # Add b_in
                    selected_indices.append(b_in)
                    cover_count[block_cover_indices[b_in]] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart local search loop after one swap
            
            if not swap_made:
                # No swap found in this pass. 
                # We could try 2-swaps, but that is O(N^2) and complex.
                # Given time constraints, we stop local search for this restart.
                break
        
        # End Local Search
        
        current_len = len(selected_indices)
        if current_len < best_len:
            best_len = current_len
            best_blocks = [list(k_blocks[i]) for i in selected_indices]
            
        # If we hit 151 or lower, we might stop early if time is tight?
        # But 151 is the known best. We aim for < 151.
        # If best_len <= 150, we can probably stop.
        if best_len <= 150:
            break
            
    # End Restart Loop
    
    if best_blocks is None:
        # Fallback: Return the first greedy result (even if not locally optimized)
        # This should not happen if the first restart completes
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        while np.any(uncovered_mask):
            gains = M @ uncovered_mask
            best_block = np.argmax(gains)
            if gains[best_block] == 0:
                break
            selected_indices.append(best_block)
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
        best_blocks = [list(k_blocks[i]) for i in selected_indices]
        
    return best_blocks
