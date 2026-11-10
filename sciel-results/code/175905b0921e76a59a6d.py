import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each (k-1)-subset, the list of t-subsets it covers
    # and for each point, the t-subsets containing it
    # Use a smarter approach: precompute for each pair of (k-1) elements, 
    # which t-subsets would be covered by adding a new point
    
    # Actually, let's precompute the coverage contribution of each possible
    # "partial block" + new point combination using bitsets
    
    # For efficiency, precompute for each (k-1)-subset S and each point x not in S,
    # the set of t-subsets that are subsets of S ∪ {x}
    # This is expensive, so instead we'll use a different strategy:
    
    # Precompute: for each point x, and for each (t-1)-subset T of other points,
    # the index of the t-subset T ∪ {x}
    # Then for a partial block B and candidate x, the new t-subsets covered are
    # those where x is in the t-subset and the other t-1 elements are in B
    
    # Let's precompute for each point x, a mapping from (t-1)-tuple to t-subset index
    # But that's still large. Instead, let's use numpy bitsets.
    
    # Strategy: For each point x, precompute a boolean array of length n_t_subsets
    # where entry i is True if x is in all_t_subsets[i]
    # And for each (t-1)-subset of points, precompute which t-subsets contain it
    
    # Better: precompute for each point x, the list of t-subsets containing x
    # and for each such t-subset, the other t-1 elements
    t_subs_containing = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for x in s:
            t_subs_containing[x].append((i, tuple(y for y in s if y != x)))
    
    # Convert to numpy arrays for fast operations
    t_subs_containing_idx = [np.array([p[0] for p in lst], dtype=np.int32) for lst in t_subs_containing]
    t_subs_containing_other = [np.array([p[1] for p in lst], dtype=np.int32) for lst in t_subs_containing]
    
    covered = np.zeros(n_t_subsets, dtype=bool)
    blocks = []
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    
    # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
    # To speed up, we maintain for each point x, how many uncovered t-subsets
    # containing x have their other t-1 elements in the current partial block
    
    # We'll build blocks one at a time using greedy point selection
    
    while not covered.all():
        if time.time() > deadline:
            break
        
        # Find first uncovered t-subset
        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            break
        first_uncovered = uncovered[0]
        S = all_t_subsets[first_uncovered]
        
        B = list(S)
        B_set = set(B)
        B_mask = np.zeros(v, dtype=bool)
        for b in B:
            B_mask[b] = True
        
        # Build the block by greedily adding points
        while len(B) < k:
            if time.time() > deadline:
                # Fill with arbitrary points
                for x in all_points:
                    if x not in B_set:
                        B.append(x)
                        B_set.add(x)
                        B_mask[x] = True
                    if len(B) == k:
                        break
                break
            
            best_x = -1
            max_gain = -1
            
            for x in all_points:
                if x in B_set:
                    continue
                
                # Compute how many uncovered t-subsets would be covered by adding x
                # These are t-subsets containing x where the other t-1 elements are in B
                idxs = t_subs_containing_idx[x]
                others = t_subs_containing_other[x]
                
                # Check which t-subsets have all their other elements in B
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
            B_set.add(best_x)
            B_mask[best_x] = True
        
        # Sort B for consistency
        B.sort()
        blocks.append(B)
        
        # Update covered array and coverage count
        for s in itertools.combinations(B, t):
            idx = subset_to_idx[s]
            covered[idx] = True
            coverage_count[idx] += 1
    
    # Local Search: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
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
    
    # Swap-based local search
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
            random.shuffle(blocks)
            
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                B = blocks[i]
                B_set = set(B)
                B_mask = np.zeros(v, dtype=bool)
                for b in B:
                    B_mask[b] = True
                
                # For each point in B, try removing it and adding another
                for b in B:
                    new_B = [x for x in B if x != b]
                    new_B_mask = B_mask.copy()
                    new_B_mask[b] = False
                    
                    best_x = -1
                    best_gain = -1
                    
                    for x in all_points:
                        if x in B_set:
                            continue
                        
                        idxs = t_subs_containing_idx[x]
                        others = t_subs_containing_other[x]
                        
                        in_new_B = new_B_mask[others]
                        valid = in_new_B.all(axis=1)
                        
                        gain = (~covered[idxs] & valid).sum()
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    
                    if best_x != -1 and best_gain > 0:
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
                            new_block = sorted(new_B + [best_x])
                            blocks[i] = new_block
                            block_t_subs[i] = [subset_to_idx[s] for s in itertools.combinations(new_block, t)]
                            
                            for idx in block_t_subs[i]:
                                coverage_count[idx] += 1
                                if coverage_count[idx] > 0:
                                    covered[idx] = True
                            
                            improved = True
                        else:
                            for idx in old_subs:
                                coverage_count[idx] += 1
    
    return blocks
