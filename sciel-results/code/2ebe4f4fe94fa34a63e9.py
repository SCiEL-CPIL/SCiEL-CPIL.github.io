import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Points for the 11-point design
    points = list(range(1, 12))
    
    # All 4-subsets of points
    all_4_subsets = list(combinations(points, 4))
    subset_to_idx = {s: i for i, s in enumerate(all_4_subsets)}
    n_4 = len(all_4_subsets) # 330
    
    # All 5-subsets of points (candidates)
    candidates = list(combinations(points, 5))
    
    # Precompute for each candidate (5-subset), the indices of the 4-subsets it contains
    cand_4_indices = []
    # Also, for each 4-subset, which candidates contain it?
    # 4-subset -> list of candidate indices
    subset_to_cands = [[] for _ in range(n_4)]
    
    for i, c in enumerate(candidates):
        indices = []
        for s in combinations(c, 4):
            idx = subset_to_idx[s]
            indices.append(idx)
            subset_to_cands[idx].append(i)
        cand_4_indices.append(indices)
        
    # Symmetry breaking: Fix the first block of the 11-point design.
    # In S(4,5,11), we can fix a block.
    # The design is unique. We can fix (1,2,3,4,5).
    first_block = (1, 2, 3, 4, 5)
    # Find index of first_block in candidates
    # candidates are sorted tuples from itertools
    # (1,2,3,4,5) is in candidates
    first_idx = candidates.index(first_block)
    
    # Initial state
    # covered_4 is a set of indices
    covered_4 = set(cand_4_indices[first_idx])
    blocks_4511 = [first_block]
    
    # Uncovered 4-subsets
    # We can track uncovered as a set of indices
    all_indices = set(range(n_4))
    uncovered = all_indices - covered_4
    
    def backtrack(covered, current_blocks, uncovered):
        if time.time() > deadline:
            return None
        if len(current_blocks) == 66:
            return current_blocks
        if not uncovered:
            return None # Should not happen if logic is correct, or return current if len matches
            
        # Heuristic: Pick the uncovered 4-subset with the fewest valid candidates
        # Valid candidate: contains the 4-subset AND is disjoint from covered
        min_count = float('inf')
        target_idx = -1
        valid_cands_for_target = []
        
        # Iterate through uncovered indices
        # Optimization: If we just pick the first one, it's O(1) selection but bad branching.
        # With "fewest candidates", we scan all uncovered.
        # |uncovered| is at most 330.
        # For each, we check candidates containing it.
        # Each 4-subset is in 7 candidates (5-subsets containing it: 11-4=7 points left, pick 1? No.
        # A 4-subset {a,b,c,d} is contained in 5-subsets {a,b,c,d,x} where x is in 1..11 \ {a,b,c,d}.
        # There are 7 such x. So 7 candidates per 4-subset.
        # So checking 330 * 7 = 2310 candidates total per step.
        # This is fast.
        
        for u_idx in uncovered:
            cands = subset_to_cands[u_idx]
            valid_cands = []
            for c_idx in cands:
                # Check if c_idx is compatible with covered
                # c_idx covers 4 4-subsets.
                # If any of those are in covered, it's invalid.
                compatible = True
                for idx in cand_4_indices[c_idx]:
                    if idx in covered:
                        compatible = False
                        break
                if compatible:
                    valid_cands.append(c_idx)
            
            if len(valid_cands) == 0:
                return None # Dead end
                
            if len(valid_cands) < min_count:
                min_count = len(valid_cands)
                target_idx = u_idx
                valid_cands_for_target = valid_cands
                
                if min_count == 1:
                    break # Can't do better than 1
                    
        if target_idx == -1:
            return None
            
        # Try each valid candidate for target_idx
        for c_idx in valid_cands_for_target:
            new_covered = set(covered)
            for idx in cand_4_indices[c_idx]:
                new_covered.add(idx)
            new_uncovered = uncovered - new_covered
            
            result = backtrack(new_covered, current_blocks + [candidates[c_idx]], new_uncovered)
            if result is not None:
                return result
                
        return None

    sol_4511 = backtrack(covered_4, blocks_4511, uncovered)
    
    if sol_4511 is None:
        # Fallback: Return a smaller packing if we fail (should not happen with good heuristic)
        # Or just return the greedy result
        covered_4 = set()
        blocks_4511 = []
        for b in candidates:
            indices = cand_4_indices[candidates.index(b)] # This is slow, precompute map?
            # Better: just use the precomputed structure
            # Re-doing greedy with indices
            pass
        
        # Simple greedy fallback
        covered_4 = set()
        blocks_4511 = []
        for i, b in enumerate(candidates):
            indices = cand_4_indices[i]
            if not any(idx in covered_4 for idx in indices):
                blocks_4511.append(b)
                covered_4.update(indices)
                if len(blocks_4511) == 66:
                    break
                    
    final_blocks = []
    for b in sol_4511:
        final_blocks.append([0] + list(b))
    for b in sol_4511:
        b_set = set(b)
        comp = [p for p in points if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks
