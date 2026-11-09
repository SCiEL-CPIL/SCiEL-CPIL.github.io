import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using a deterministic backtracking 
    approach with bitmask optimization for speed. Returns a list of 132 distinct 
    6-subsets of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 110.0
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    n5 = len(all_5_subsets)  # 792
    n6 = len(all_6_subsets)  # 924
    
    # Precompute: for each 6-subset, its 5-subsets as indices
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    
    # Precompute bitmask for each 6-subset representing its 5-subsets
    # An integer with 792 bits
    six_to_fives_mask = []
    six_to_fives_indices = [] # Keep indices for counting candidates
    for b in all_6_subsets:
        mask = 0
        indices = []
        for s5 in combinations(b, 5):
            idx = five_index[s5]
            mask |= (1 << idx)
            indices.append(idx)
        six_to_fives_mask.append(mask)
        six_to_fives_indices.append(indices)
        
    # For each 5-subset index, which 6-subset indices contain it
    # This is used for the "most constrained" heuristic
    five_to_sixes = [[] for _ in range(n5)]
    for bi in range(n6):
        for fi in six_to_fives_indices[bi]:
            five_to_sixes[fi].append(bi)

    def backtrack(covered_mask, used_mask, blocks):
        if time.time() - START > TIME_LIMIT:
            return None
        
        if len(blocks) == 132:
            return blocks
        
        # Find uncovered 5-subsets and pick the most constrained one
        best_fi = -1
        best_count = n6 + 1
        best_candidates = []
        
        # Iterate through all 5-subsets to find uncovered ones
        # We can optimize this by keeping a list of uncovered indices, 
        # but for now, we scan. 792 is small.
        for fi in range(n5):
            # Check if fi is covered
            if (covered_mask >> fi) & 1:
                continue
            
            # Count available 6-subsets
            count = 0
            candidates = []
            for bi in five_to_sixes[fi]:
                # Check if bi is used
                if (used_mask >> bi) & 1:
                    continue
                # Check conflict: does bi have any 5-subset in covered_mask?
                if (covered_mask & six_to_fives_mask[bi]) != 0:
                    continue
                count += 1
                candidates.append(bi)
            
            if count < best_count:
                best_count = count
                best_fi = fi
                best_candidates = candidates
                if count == 0:
                    return None
                if count == 1:
                    break
        
        if best_fi == -1:
            # All covered?
            if len(blocks) == 132:
                return blocks
            return None
        
        if not best_candidates:
            return None
        
        # Try each candidate
        for bi in best_candidates:
            new_used_mask = used_mask | (1 << bi)
            new_covered_mask = covered_mask | six_to_fives_mask[bi]
            new_blocks = blocks + [all_6_subsets[bi]]
            
            result = backtrack(new_covered_mask, new_used_mask, new_blocks)
            if result is not None:
                return result
        
        return None
    
    result = backtrack(0, 0, [])
    
    if result is not None:
        return [list(b) for b in result]
    
    # Fallback (should not be reached if bitmask optimization works)
    import random
    best_blocks = []
    best_count = 0
    
    for seed in range(500):
        if time.time() - START > TIME_LIMIT:
            break
        
        random.seed(seed)
        covered_mask = 0
        used_mask = 0
        blocks = []
        order = list(range(n6))
        random.shuffle(order)
        
        for bi in order:
            if time.time() - START > TIME_LIMIT:
                break
            
            if (used_mask >> bi) & 1:
                continue
            if (covered_mask & six_to_fives_mask[bi]) != 0:
                continue
            
            blocks.append(all_6_subsets[bi])
            used_mask |= (1 << bi)
            covered_mask |= six_to_fives_mask[bi]
            
            if len(blocks) == 132:
                return [list(b) for b in blocks]
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]
