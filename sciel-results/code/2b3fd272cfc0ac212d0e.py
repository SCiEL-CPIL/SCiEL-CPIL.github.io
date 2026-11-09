import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using a deterministic Exact Cover (Algorithm X) solver.
    Returns a list of 132 distinct 6-subsets of 0..11.
    """
    START = time.time()
    TIME_LIMIT = 110.0
    
    # Points are 0..11
    points = list(range(12))
    
    # Generate all 5-subsets (constraints) and 6-subsets (blocks)
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    n5 = len(all_5_subsets) # 792
    n6 = len(all_6_subsets) # 924
    
    # Map 5-subsets to indices
    five_index = {s5: i for i, s5 in enumerate(all_5_subsets)}
    
    # Precompute for each 6-subset, the list of 5-subset indices it covers
    # and for each 5-subset, the list of 6-subset indices that cover it
    six_to_fives = []
    five_to_sixes = [[] for _ in range(n5)]
    
    for b in all_6_subsets:
        fives = [five_index[s5] for s5 in combinations(b, 5)]
        six_to_fives.append(fives)
        for fi in fives:
            five_to_sixes[fi].append(all_6_subsets.index(b)) # Using index lookup is slow, let's use enumerate
    
    # Re-do with enumerate for efficiency
    five_to_sixes = [[] for _ in range(n5)]
    for bi, b in enumerate(all_6_subsets):
        fives = [five_index[s5] for s5 in combinations(b, 5)]
        six_to_fives[bi] = fives
        for fi in fives:
            five_to_sixes[fi].append(bi)

    # Algorithm X (Dancing Links or bitmask based)
    # We want to select a set of 6-subsets such that every 5-subset is covered exactly once.
    # This is an exact cover problem.
    
    # State:
    # covered: set of 5-subset indices that are already covered
    # used: set of 6-subset indices that are already used
    
    # Optimization: Use bitsets if possible, but sets are fine for 792/924.
    # Heuristic: Choose the 5-subset with the fewest available covering 6-subsets.
    
    def backtrack(covered, used, blks):
        if time.time() - START > TIME_LIMIT:
            return None
        
        if len(covered) == n5:
            return blks
        
        # Find the 5-subset with the minimum number of valid covering 6-subsets
        min_count = n6 + 1
        best_fi = -1
        best_candidates = []
        
        for fi in range(n5):
            if fi in covered:
                continue
            
            count = 0
            candidates = []
            for bi in five_to_sixes[fi]:
                if bi in used:
                    continue
                # Check if this block conflicts with already covered 5-subsets
                conflict = False
                for fi2 in six_to_fives[bi]:
                    if fi2 in covered:
                        conflict = True
                        break
                if not conflict:
                    count += 1
                    candidates.append(bi)
            
            if count < min_count:
                min_count = count
                best_fi = fi
                best_candidates = candidates
                if count == 0:
                    return None
                if count == 1:
                    break # Can't get better than 1
            
        if best_fi == -1:
            # All covered?
            return blks if len(covered) == n5 else None
            
        if not best_candidates:
            return None
        
        # Try each candidate
        for bi in best_candidates:
            new_used = used | {bi}
            new_covered = covered | set(six_to_fives[bi])
            new_blks = blks + [all_6_subsets[bi]]
            
            result = backtrack(new_covered, new_used, new_blks)
            if result is not None:
                return result
        
        return None

    result = backtrack(set(), set(), [])
    
    if result is not None:
        return result
    
    # Fallback if search fails (should not happen for S(5,6,12) which is unique)
    # Return empty or partial if timeout
    return []
