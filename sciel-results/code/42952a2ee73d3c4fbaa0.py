import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    # Map 5-subset to index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    num_5_subsets = len(all_5_subsets) # 792
    
    # Precompute all 6-subsets and their constituent 5-subsets indices
    all_6_subsets = list(combinations(points, 6))
    block_5s = []
    for b in all_6_subsets:
        idxs = [subset_to_idx[s] for s in combinations(b, 5)]
        block_5s.append(idxs)
        
    # We want to find 132 blocks.
    # We use backtracking.
    # To leverage symmetry, we can fix the first block.
    # The group M12 is 5-transitive. 
    # Any block can be mapped to {0,1,2,3,4,5} by an automorphism? 
    # Yes, because it's 4-transitive on blocks? No, it's 5-transitive on points.
    # So we can fix the first block to be {0,1,2,3,4,5}.
    
    target_block = (0, 1, 2, 3, 4, 5)
    target_idx = all_6_subsets.index(target_block)
    
    covered_mask = [False] * num_5_subsets
    blocks = []
    
    # Add the first block
    for idx in block_5s[target_idx]:
        covered_mask[idx] = True
    blocks.append(list(target_block))
    
    # Recursive search
    # We need to find 131 more blocks.
    # To speed up, we can order the candidate blocks.
    
    # A simple heuristic: choose the next block that covers the most "rare" subsets?
    # Or just iterate. With 924 blocks, and pruning, it should be fast.
    
    def backtrack(remaining_blocks_needed, available_block_indices):
        if time.time() > deadline:
            return None
            
        if remaining_blocks_needed == 0:
            return list(blocks)
            
        # Heuristic: Pick a 5-subset that is not covered yet?
        # No, we just need to add blocks.
        # We can iterate through available blocks.
        
        # Optimization: If we have time, we can try to be smart.
        # But for 132 blocks, simple iteration with pruning might be enough.
        
        for i in range(len(available_block_indices)):
            b_idx = available_block_indices[i]
            b_5s = block_5s[b_idx]
            
            # Check if any 5-subset is already covered
            conflict = False
            for s_idx in b_5s:
                if covered_mask[s_idx]:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Add block
            blocks.append(list(all_6_subsets[b_idx]))
            for s_idx in b_5s:
                covered_mask[s_idx] = True
                
            # Prepare next available list (remove current and any conflicting)
            new_available = []
            for j in range(i + 1, len(available_block_indices)):
                next_idx = available_block_indices[j]
                next_5s = block_5s[next_idx]
                ok = True
                for s_idx in next_5s:
                    if covered_mask[s_idx]:
                        ok = False
                        break
                if ok:
                    new_available.append(next_idx)
                    
            res = backtrack(remaining_blocks_needed - 1, new_available)
            if res is not None:
                return res
                
            # Undo
            blocks.pop()
            for s_idx in b_5s:
                covered_mask[s_idx] = False
                
        return None

    # Initial available blocks: all except the first one
    initial_available = [i for i in range(len(all_6_subsets)) if i != target_idx]
    
    # We need 131 more blocks
    result = backtrack(131, initial_available)
    
    if result is None:
        # Fallback to greedy if backtracking fails (shouldn't happen)
        covered, blocks = set(), []
        for b in combinations(range(12), 6):
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
        
    return result
