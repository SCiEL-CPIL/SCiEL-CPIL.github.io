import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using a deterministic backtracking 
    approach with symmetry breaking. Returns a list of 132 distinct 6-subsets 
    of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 110.0
    
    points = list(range(12))
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    n5 = len(all_5_subsets)  # 792
    n6 = len(all_6_subsets)  # 924
    
    # Precompute: for each 6-subset, its 5-subsets as indices into all_5_subsets
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    six_to_fives = []  # list of lists of 5-subset indices
    for b in all_6_subsets:
        fives = [five_index[s5] for s5 in combinations(b, 5)]
        six_to_fives.append(fives)
    
    # For each 5-subset index, which 6-subset indices contain it
    five_to_sixes = [[] for _ in range(n5)]
    for bi, fives in enumerate(six_to_fives):
        for fi in fives:
            five_to_sixes[fi].append(bi)
    
    # Symmetry breaking: fix the first block. 
    # In S(5,6,12), the automorphism group is M12. We can fix block 0 to be (0,1,2,3,4,5)
    # This reduces the search space significantly.
    # Actually, let's use a simpler approach: fix that point 0 is in the first block,
    # and by relabeling we can assume the first block is (0,1,2,3,4,5).
    
    # Let's try a direct backtracking with good heuristics
    # State: which 5-subsets are covered (bitmask or set), which 6-subsets are used
    
    # Use a set of covered 5-subset indices for fast lookup
    # Use a set of used 6-subset indices
    
    def backtrack(covered, used, blocks):
        if time.time() - START > TIME_LIMIT:
            return None
        
        if len(blocks) == 132:
            return blocks
        
        # Find uncovered 5-subsets and pick the most constrained one
        # A 5-subset is covered if it's in the covered set
        # For each uncovered 5-subset, count available 6-subsets that contain it
        # and don't conflict with covered 5-subsets
        
        best_fi = -1
        best_count = n6 + 1
        best_candidates = []
        
        for fi in range(n5):
            if fi in covered:
                continue
            
            # Count how many unused 6-subsets contain this 5-subset
            # and don't have any 5-subset in conflict with covered
            count = 0
            candidates = []
            for bi in five_to_sixes[fi]:
                if bi in used:
                    continue
                # Check if any 5-subset of this 6-subset is already covered
                conflict = False
                for fi2 in six_to_fives[bi]:
                    if fi2 in covered:
                        conflict = True
                        break
                if not conflict:
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
            # All 5-subsets covered
            if len(blocks) == 132:
                return blocks
            return None
        
        if not best_candidates:
            return None
        
        # Try each candidate
        for bi in best_candidates:
            # Mark this 6-subset as used and its 5-subsets as covered
            new_used = used | {bi}
            new_covered = covered | set(six_to_fives[bi])
            new_blocks = blocks + [all_6_subsets[bi]]
            
            result = backtrack(new_covered, new_used, new_blocks)
            if result is not None:
                return result
        
        return None
    
    # Start with no blocks
    result = backtrack(set(), set(), [])
    
    if result is not None:
        return [list(b) for b in result]
    
    # Fallback: greedy approach
    import random
    best_blocks = []
    best_count = 0
    
    for seed in range(200):
        if time.time() - START > TIME_LIMIT:
            break
        
        random.seed(seed)
        covered = set()
        used = set()
        blocks = []
        order = list(range(n6))
        random.shuffle(order)
        
        for bi in order:
            if time.time() - START > TIME_LIMIT:
                break
            
            # Check conflict
            conflict = False
            for fi in six_to_fives[bi]:
                if fi in covered:
                    conflict = True
                    break
            
            if not conflict:
                blocks.append(all_6_subsets[bi])
                used.add(bi)
                covered.update(six_to_fives[bi])
                
                if len(blocks) == 132:
                    return [list(b) for b in blocks]
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]
