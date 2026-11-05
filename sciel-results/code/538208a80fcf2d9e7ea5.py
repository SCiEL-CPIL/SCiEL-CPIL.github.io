import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for 120s limit
    
    # Precompute all t-subsets and map them to IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point, the list of t-subsets it belongs to
    # This helps in calculating gain
    # Actually, for a partial block B, adding x covers subsets {x} U C(B, t-1)
    
    best_blocks = None
    best_count = float('inf')
    
    # Try a few random restarts or different greedy strategies
    # Strategy 1: Standard Greedy
    # Strategy 2: Greedy with random tie-breaking
    
    for restart in range(10): # Cap restarts, break if time up
        if time.time() - start_time > timeout:
            break
            
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        # Shuffle the initial order of subsets to break symmetry in tie-breaking?
        # Or just pick the first one.
        # To make restarts effective, we can randomize the priority of points when gains are equal.
        
        while uncovered:
            if time.time() - start_time > timeout:
                break
                
            # Pick the first uncovered subset (or one with max "weight" if we had weights)
            # For simplicity, pick min ID
            target_subset_id = min(uncovered)
            target_subset = all_t_subsets[target_subset_id]
            
            block = list(target_subset)
            block_set = set(block)
            
            while len(block) < k:
                if time.time() - start_time > timeout:
                    break
                    
                best_gain = -1
                best_points = []
                
                # Candidates
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: how many uncovered subsets are formed by x + 4 points from block
                    gain = 0
                    # C(len(block), t-1)
                    for rest in itertools.combinations(block, t - 1):
                        # Create the t-subset
                        # rest is a tuple of t-1 elements
                        # We need to add x and sort
                        # Optimization: precompute sorted tuples?
                        # Or just check membership in subset_to_id
                        new_subset = tuple(sorted(list(rest) + [x]))
                        if new_subset in subset_to_id:
                            sid = subset_to_id[new_subset]
                            if sid in uncovered:
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                        
                if not best_points:
                    # Should not happen unless k < t or something, but v > k
                    break
                    
                # Randomly pick from best points to allow different paths
                x = random.choice(best_points)
                block.append(x)
                block_set.add(x)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                sid = subset_to_id[sub]
                if sid in uncovered:
                    uncovered.remove(sid)
                    
        if uncovered:
            # Incomplete due to timeout
            continue
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        if best_count <= 335: # Known best
            break
            
    if best_blocks is None:
        # Fallback: return empty or a dummy valid cover if we failed completely?
        # But the problem guarantees a solution exists.
        # If we timed out, we should return the best partial? No, must be valid.
        # The loop above only updates best_blocks if uncovered is empty.
        # If we never finished a run, we must run one fast, non-random greedy to ensure validity.
        
        # Fast fallback greedy (deterministic)
        uncovered = set(range(n_t_subsets))
        blocks = []
        while uncovered:
            target_subset_id = min(uncovered)
            target_subset = all_t_subsets[target_subset_id]
            block = list(target_subset)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_subset = tuple(sorted(list(rest) + [x]))
                        if new_subset in subset_to_id:
                            sid = subset_to_id[new_subset]
                            if sid in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                sid = subset_to_id[sub]
                if sid in uncovered:
                    uncovered.remove(sid)
        best_blocks = blocks
        best_count = len(blocks)

    return best_blocks
