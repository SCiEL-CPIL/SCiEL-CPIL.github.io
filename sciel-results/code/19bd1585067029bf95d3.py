import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    block_t_indices = [] 
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_subs_in_block = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[s] for s in t_subs_in_block], dtype=np.int32)
        block_t_indices.append(indices)
        for idx in indices:
            t_subset_to_blocks[idx].append(b_idx)
    
    # Precompute bitmask representation for fast gain calculation
    # Each block maps to a bitmask of which t-subsets it covers
    # Since num_t_subsets can be large, we use a sparse representation instead
    # But for C(19,8,5), num_t_subsets = 11628, which fits in a few 64-bit words
    # Let's use numpy arrays for the t-indices (already done)
    
    # Generate structured blocks using finite field constructions
    def generate_structured_blocks():
        """Generate blocks using partition into groups and affine transformations."""
        structured_blocks = []
        seen = set()
        
        # Partition 19 points into groups: 7, 6, 6
        # Group 0: points 0-6 (size 7, use F_7)
        # Group 1: points 7-12 (size 6, use F_5 + 1 extra)
        # Group 2: points 13-18 (size 6, use F_5 + 1 extra)
        
        group_sizes = [7, 6, 6]
        group_starts = [0, 7, 13]
        
        # For each combination of how many points to take from each group
        # such that total = k = 8
        for n0 in range(min(7, k) + 1):
            for n1 in range(min(6, k - n0) + 1):
                n2 = k - n0 - n1
                if n2 < 0 or n2 > 6:
                    continue
                
                # Generate all combinations of sizes n0, n1, n2 from each group
                group0_combos = list(itertools.combinations(range(group_starts[0], group_starts[0] + group_sizes[0]), n0))
                group1_combos = list(itertools.combinations(range(group_starts[1], group_starts[1] + group_sizes[1]), n1))
                group2_combos = list(itertools.combinations(range(group_starts[2], group_starts[2] + group_sizes[2]), n2))
                
                for c0 in group0_combos:
                    for c1 in group1_combos:
                        for c2 in group2_combos:
                            block = tuple(sorted(list(c0) + list(c1) + list(c2)))
                            if block not in seen:
                                seen.add(block)
                                structured_blocks.append(block)
        
        return structured_blocks
    
    # Generate structured blocks
    structured_blocks = generate_structured_blocks()
    structured_block_indices = []
    for block in structured_blocks:
        if block in t_subset_to_idx:
            pass
        # Find the index of this block in all_k_blocks
        # This is inefficient, so let's create a lookup
        pass
    
    # Create a faster lookup: block tuple -> index
    block_to_idx = {block: i for i, block in enumerate(all_k_blocks)}
    
    structured_block_indices = []
    for block in structured_blocks:
        if block in block_to_idx:
            structured_block_indices.append(block_to_idx[block])
    
    best_block_indices = None
    best_score = float('inf')
    
    restart = 0
    while True:
        if time.time() > deadline - 10.0:
            break
            
        random.seed(restart)
        
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        # Phase 1: Add structured blocks (random subset to vary coverage)
        if structured_block_indices:
            # Randomly select a subset of structured blocks
            num_to_add = min(len(structured_block_indices), random.randint(50, 200))
            selected_structured = random.sample(structured_block_indices, num_to_add)
            for b_idx in selected_structured:
                block_indices.append(b_idx)
                idxs = block_t_indices[b_idx]
                coverage[idxs] += 1
        
        # Phase 2: Greedy repair for uncovered subsets
        uncovered_count = np.count_nonzero(coverage == 0)
        
        while uncovered_count > 0:
            # Find an uncovered subset
            while True:
                u_idx = random.randint(0, num_t_subsets - 1)
                if coverage[u_idx] == 0:
                    break
            
            # Get candidates from K random uncovered subsets
            K = 15
            candidate_block_set = set()
            
            for b_idx in t_subset_to_blocks[u_idx]:
                candidate_block_set.add(b_idx)
                
            for _ in range(K - 1):
                while True:
                    u_idx2 = random.randint(0, num_t_subsets - 1)
                    if coverage[u_idx2] == 0:
                        break
                for b_idx in t_subset_to_blocks[u_idx2]:
                    candidate_block_set.add(b_idx)
            
            if not candidate_block_set:
                break
            
            # Vectorized Gain Calculation
            cand_list = list(candidate_block_set)
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            
            # Calculate gain: number of newly covered t-subsets
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            
            best_pos = np.argmax(gains)
            best_block_idx = cand_list[best_pos]
            
            # Add block
            block_indices.append(best_block_idx)
            
            # Update coverage
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
            
            uncovered_count = np.count_nonzero(coverage == 0)
                
        # Reduction Phase
        changed = True
        current_indices = block_indices[:]
        
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                t_idxs = block_t_indices[b_idx]
                if np.any(coverage[t_idxs] == 1):
                    new_indices.append(b_idx)
                else:
                    coverage[t_idxs] -= 1
                    changed = True
            current_indices = new_indices
            
        # Update best
        if len(current_indices) < best_score:
            best_score = len(current_indices)
            best_block_indices = current_indices[:]
            
        restart += 1
        
    if best_block_indices is None:
        # Fallback: pure greedy
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        
        while True:
            uncovered_count = np.count_nonzero(coverage == 0)
            if uncovered_count == 0:
                break
            
            u_idx = random.randint(0, num_t_subsets - 1)
            while coverage[u_idx] != 0:
                u_idx = random.randint(0, num_t_subsets - 1)
            
            candidate_block_set = set(t_subset_to_blocks[u_idx])
            cand_list = list(candidate_block_set)
            cand_t_arr = np.vstack([block_t_indices[b] for b in cand_list])
            gains = np.sum(coverage[cand_t_arr] == 0, axis=1)
            best_block_idx = cand_list[np.argmax(gains)]
            
            block_indices.append(best_block_idx)
            idxs = block_t_indices[best_block_idx]
            coverage[idxs] += 1
        
        best_block_indices = block_indices
    
    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_block_indices]
