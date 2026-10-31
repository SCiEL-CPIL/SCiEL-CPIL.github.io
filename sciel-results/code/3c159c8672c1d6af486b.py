import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    # block_to_tsubs[block_idx] = list of t_subset indices
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
    
    # Precompute inverse: which blocks cover each t-subset?
    # tsub_to_blocks[tsub_idx] = numpy array of block indices
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for fast indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Construction
    # We want to minimize the number of blocks.
    # Standard greedy: pick the block that covers the most *uncovered* t-subsets.
    
    # Initial state: all t-subsets uncovered
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    # Gain for each block: number of currently uncovered t-subsets it covers
    # Initially, every block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            # If we time out during greedy, we have an invalid cover.
            # However, for v=19, k=7, t=4, greedy is very fast.
            # If it doesn't finish, we might try to finish it quickly?
            # No, better to just break and return what we have? 
            # But invalid score is 0. 
            # The greedy loop for this size is ~150 iterations.
            # Each iteration is O(num_blocks + updates).
            # Updates are sparse.
            # It should finish in < 1s.
            break
        
        # Find block with max gain
        b = np.argmax(gains)
        
        if gains[b] == 0:
            # No block can cover any more uncovered subsets?
            # This implies uncovered is not empty but no block intersects it?
            # This shouldn't happen if the covering number is finite and we have all blocks.
            # Wait, if uncovered is not empty, there must be some block covering it.
            # Unless we ran out of blocks? No, we have all k-subsets.
            # So gains[b] > 0 if uncovered.any() is True.
            break
        
        selected_block_indices.append(b)
        
        # Update uncovered and gains
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                # Decrease gain for all blocks that cover this t-subset
                # We only care about blocks that might be selected in the future
                gains[tsub_to_blocks_np[s_idx]] -= 1
                
    if not uncovered.any():
        # 3. Local Optimization (Redundancy Removal + Swaps)
        # Goal: Reduce the number of blocks in selected_block_indices.
        
        # Build cover_count: how many selected blocks cover each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Helper: Check if a block is redundant
        def is_redundant(b_idx):
            # A block is redundant if all t-subsets it covers are covered by >= 2 blocks
            # i.e., removing it leaves cover_count[s] >= 1 for all s in block
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # Helper: Find a replacement block for a set of unique t-subsets
        # unique_tsubs: list of t-subset indices that are ONLY covered by the block to be removed
        # We need a block that covers all of them, and is NOT already in selected_block_indices
        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None # Should not happen for non-redundant block? 
                           # Actually, if unique_tsubs is empty, it's redundant.
            
            # Get candidates from the first unique t-subset
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all other unique t-subsets
                covers_all = True
                for s in unique_tsubs[1:]:
                    # Check if c covers s
                    # We can precompute a set of t-subsets for each block? 
                    # Or just check if s is in block_to_tsubs[c]
                    # Since block_to_tsubs[c] is a list, membership is O(n).
                    # Better: use a set for fast lookup? 
                    # We didn't precompute sets for all blocks to save memory/time.
                    # But we can check: is s in the list of t-subsets covered by c?
                    # Let's optimize: 
                    # We know c covers first_s.
                    # We need to check if c covers s for all other s in unique_tsubs.
                    # We can do this by checking if the intersection of t-subsets covered by c 
                    # and unique_tsubs is equal to unique_tsubs.
                    
                    # Simpler: just iterate unique_tsubs and check membership in block_to_tsubs[c]
                    # To make this fast, we can precompute a set of t-subsets for each block?
                    # That's C(19,7) * C(7,4) = 50388 * 35 ~ 1.7M entries. 
                    # Sets of ints are okay.
                    # But we already have block_to_tsubs as lists.
                    # Let's just do a linear scan for now, unique_tsubs is small (<= 35).
                    
                    # Optimization: Precompute block_to_tsub_set?
                    # Let's do it lazily or globally?
                    # Global precomputation of sets is expensive in memory but fast in lookup.
                    # Let's assume linear scan is okay for now, or we can use np.isin?
                    # np.isin(unique_tsubs, block_to_tsubs[c])
                    
                    if s not in block_to_tsubs[c]:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            # Pick a random valid candidate to allow exploration
            return random.choice(valid_candidates)
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            # Find all redundant blocks
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                # Remove them
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart loop to check for new redundancies
            
            # Step B: Swap
            # Calculate unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_block_indices:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            # Find block with minimum unique count (> 0)
            # If min unique count is 0, it's redundant (handled above)
            min_unique = min(unique_counts.values())
            
            # Candidates for removal: blocks with min_unique
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            
            # Shuffle to break symmetry
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                
                # Find a replacement
                # Exclude all currently selected blocks
                exclude_set = set(selected_block_indices)
                
                b_in = find_replacement(s_uniq, exclude_set)
                
                if b_in is not None:
                    # Perform swap
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    # Update cover counts
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
            
            if not swap_made:
                # No swap found with 1-block swap.
                # We could try 2-block swaps, but it's complex.
                # Or we can stop.
                break
        
        # Convert indices back to blocks
        result = [list(all_k_subsets[i]) for i in selected_block_indices]
        return result
    else:
        # Greedy didn't finish? Return invalid?
        # This should not happen.
        # Return empty or partial?
        # To be safe, return empty list which is invalid, score 0.
        return []

# Helper to speed up find_replacement: Precompute sets for block_to_tsubs?
# The linear scan in find_replacement might be a bottleneck if called often.
# Let's optimize find_replacement by precomputing a set for each block's t-subsets.
# We can do this inside construct if needed, or globally.
# Since construct is called once, we can add it there.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    block_to_tsubs = []
    block_to_tsubs_set = [] # For fast membership in find_replacement
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
    
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
        
        selected_block_indices.append(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                gains[tsub_to_blocks_np[s_idx]] -= 1
                
    if not uncovered.any():
        # 3. Local Optimization
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all unique_tsubs
                # Using set intersection is fast
                # unique_tsubs is a list, convert to set? Or just check membership in block_to_tsubs_set[c]
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_block_indices:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                exclude_set = set(selected_block_indices)
                
                b_in = find_replacement(s_uniq, exclude_set)
                
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break
            
            if not swap_made:
                break
        
        result = [list(all_k_subsets[i]) for i in selected_block_indices]
        return result
    else:
        return []
