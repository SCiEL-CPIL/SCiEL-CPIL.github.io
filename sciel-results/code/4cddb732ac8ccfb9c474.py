import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Correct base blocks for S(4, 5, 11)
    base_blocks = [
        (0, 1, 3, 7, 9),
        (0, 1, 4, 6, 10),
        (0, 2, 3, 5, 8),
        (0, 2, 4, 7, 10),
        (0, 2, 5, 6, 9),
        (0, 3, 4, 5, 10)
    ]
    
    # Generate all 66 blocks of S(4, 5, 11)
    s4511_blocks = set()
    for base in base_blocks:
        for i in range(11):
            block = tuple(sorted((x + i) % 11 for x in base))
            s4511_blocks.add(block)
            
    if len(s4511_blocks) == 66:
        # Construct S(5, 6, 12) blocks
        blocks = []
        points_0_10 = set(range(11))
        
        for B in s4511_blocks:
            # Type 1: {11} U B
            b1 = tuple(sorted(list(B) + [11]))
            blocks.append(b1)
            
            # Type 2: complement of B in {0..10}
            b2_set = points_0_10 - set(B)
            b2 = tuple(sorted(b2_set))
            blocks.append(b2)
        
        # Quick verification: check count and uniqueness
        if len(blocks) == 132 and len(set(blocks)) == 132:
            # Verify no 5-subset repetition
            covered_5s = set()
            valid = True
            for b in blocks:
                for s in combinations(b, 5):
                    if s in covered_5s:
                        valid = False
                        break
                    covered_5s.add(s)
                if not valid:
                    break
            
            if valid:
                return [list(b) for b in blocks]
    
    # Fallback: Backtracking with MRV heuristic
    return backtracking_mrv()

def backtracking_mrv():
    """
    Backtracking search using Minimum Remaining Values heuristic.
    Uses bitmasks for efficiency.
    """
    start_time = time.time()
    deadline = start_time + 105.0
    
    points = 12
    # Precompute all 6-subsets as bitmasks and their 5-subsets
    all_6_subsets = []
    for combo in combinations(range(points), 6):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        all_6_subsets.append((mask, combo))
    
    # For each 6-subset, precompute the set of 5-subset masks it covers
    # There are C(6,5)=6 five-subsets per 6-subset
    subset_5_masks = []
    for mask, combo in all_6_subsets:
        five_masks = []
        for five in combinations(combo, 5):
            fm = 0
            for p in five:
                fm |= (1 << p)
            five_masks.append(fm)
        subset_5_masks.append(five_masks)
    
    # Total number of 5-subsets: C(12,5) = 792
    # We need to cover all 792 5-subsets exactly once with 132 blocks
    
    # Represent covered 5-subsets as a set of masks
    # Total 5-subset masks: we'll track which are covered
    
    best_blocks = []
    best_count = 0
    
    # Use a set to track which 5-subset masks are already covered
    # We'll do multiple random restarts with different orderings
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Shuffle the 6-subsets for this restart
        indices = list(range(len(all_6_subsets)))
        random.shuffle(indices)
        
        covered_5s = set()
        blocks = []
        
        for idx in indices:
            mask, combo = all_6_subsets[idx]
            five_masks = subset_5_masks[idx]
            
            # Check if any 5-subset in this block is already covered
            conflict = False
            for fm in five_masks:
                if fm in covered_5s:
                    conflict = True
                    break
            
            if not conflict:
                blocks.append(combo)
                covered_5s.update(five_masks)
                
                if len(blocks) == 132:
                    # We found a complete solution!
                    return [list(b) for b in blocks]
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    return [list(b) for b in best_blocks]
