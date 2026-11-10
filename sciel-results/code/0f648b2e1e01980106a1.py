import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precomputation
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point x, the indices of t-subsets containing x
    # and the other t-1 elements of those subsets
    subset_indices_by_point = {}
    other_elements_by_point = {}
    
    for x in all_points:
        idxs = []
        others = []
        for i, s in enumerate(all_t_subsets):
            if x in s:
                idxs.append(i)
                others.append(tuple(y for y in s if y != x))
        subset_indices_by_point[x] = np.array(idxs, dtype=np.int32)
        other_elements_by_point[x] = np.array(others, dtype=np.int8)
        
    # Covered array
    covered = np.zeros(n_t_subsets, dtype=bool)
    blocks = []
    
    # Greedy Construction with better gain computation
    while not covered.all():
        if time.time() > deadline:
            break
            
        # Find first uncovered subset
        uncovered_idx = np.where(~covered)[0]
        if len(uncovered_idx) == 0:
            break
        first_uncovered_idx = uncovered_idx[0]
        S = all_t_subsets[first_uncovered_idx]
        
        B = list(S)
        
        # Build the block
        while len(B) < k:
            if time.time() > deadline:
                for x in all_points:
                    if x not in B:
                        B.append(x)
                    if len(B) == k:
                        break
                break
            
            best_x = -1
            max_gain = -1
            B_set = set(B)
            B_mask = np.zeros(v, dtype=bool)
            for b in B:
                B_mask[b] = True
                
            for x in all_points:
                if x in B_set:
                    continue
                
                idxs = subset_indices_by_point[x]
                others = other_elements_by_point[x]
                
                # Check if others are in B
                in_B = B_mask[others]
                valid = in_B.all(axis=1)
                
                # Count uncovered valid subsets
                gain = (~covered[idxs] & valid).sum()
                
                if gain > max_gain:
                    max_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
            B.append(best_x)
        
        # Sort B for consistency
        B.sort()
        blocks.append(B)
        
        # Update covered array
        for s in itertools.combinations(B, t):
            idx = subset_to_idx[s]
            covered[idx] = True
            
    # Local Search: Remove redundant blocks
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    
    for B in blocks:
        for s in itertools.combinations(B, t):
            idx = subset_to_idx[s]
            coverage_count[idx] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            B = blocks[i]
            redundant = True
            for s in itertools.combinations(B, t):
                idx = subset_to_idx[s]
                if coverage_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                for s in itertools.combinations(B, t):
                    idx = subset_to_idx[s]
                    coverage_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break
    
    # Swap-based local search: try to replace one point in a block with another
    # to improve coverage while maintaining validity
    if time.time() < deadline - 10:
        # Precompute which t-subsets each block covers
        block_t_subs = []
        for B in blocks:
            subs = list(itertools.combinations(B, t))
            block_t_subs.append([subset_to_idx[s] for s in subs])
        
        # Try swap moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(blocks)  # Randomize order for diversity
            
            for i, B in enumerate(blocks):
                if time.time() > deadline - 5:
                    break
                B_set = set(B)
                B_mask = np.zeros(v, dtype=bool)
                for b in B:
                    B_mask[b] = True
                
                # For each point in B, try removing it and adding another
                for b in B:
                    new_B = [x for x in B if x != b]
                    new_B_mask = B_mask.copy()
                    new_B_mask[b] = False
                    
                    # Find best replacement
                    best_x = -1
                    best_gain = -1
                    
                    for x in all_points:
                        if x in B_set:
                            continue
                        
                        # Compute gain: how many t-subsets become covered
                        # that weren't covered before
                        idxs = subset_indices_by_point[x]
                        others = other_elements_by_point[x]
                        
                        in_new_B = new_B_mask[others]
                        valid = in_new_B.all(axis=1)
                        
                        gain = (~covered[idxs] & valid).sum()
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x != -1 and best_gain > 0:
                        # Check if this swap maintains validity
                        # (all t-subsets still covered)
                        old_subs = block_t_subs[i]
                        
                        # Subtract coverage of old block
                        for idx in old_subs:
                            coverage_count[idx] -= 1
                        
                        # Check if any t-subset becomes uncovered
                        valid_swap = True
                        for idx in old_subs:
                            if coverage_count[idx] == 0:
                                valid_swap = False
                                break
                        
                        if valid_swap:
                            # Apply swap
                            blocks[i] = sorted(new_B + [best_x])
                            block_t_subs[i] = [subset_to_idx[s] for s in itertools.combinations(blocks[i], t)]
                            
                            # Add coverage of new block
                            for idx in block_t_subs[i]:
                                coverage_count[idx] += 1
                            
                            # Update covered array
                            for idx in block_t_subs[i]:
                                if coverage_count[idx] > 0:
                                    covered[idx] = True
                            
                            improved = True
                        else:
                            # Revert coverage count
                            for idx in old_subs:
                                coverage_count[idx] += 1
    
    return blocks
