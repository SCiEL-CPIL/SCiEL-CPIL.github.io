import time
from itertools import combinations

def _solve_s4511():
    # Returns list of 66 blocks (tuples) for S(4,5,11)
    points = list(range(11))
    target_blocks = 66
    # Precompute all 4-subsets
    all_4subs = list(combinations(points, 4))
    covered = set()
    blocks = []
    
    # Symmetry breaking: first block is (0,1,2,3,4)
    first_block = (0, 1, 2, 3, 4)
    for sub in combinations(first_block, 4):
        covered.add(sub)
    blocks.append(first_block)
    
    def backtrack():
        if len(blocks) == target_blocks:
            return True
        if time.time() > start_time + 10: # Safety
            return False
            
        # Find smallest uncovered 4-subset
        current_smallest = None
        for s in all_4subs:
            if s not in covered:
                current_smallest = s
                break
        
        if current_smallest is None:
            return False # All covered but not enough blocks? Should not happen
            
        # Try all 5-subsets containing current_smallest
        # 5-subset B must contain the 4 points of current_smallest
        # The 5th point can be any of the other 7 points (11-4=7)
        # But we must ensure the new 4-subsets don't conflict
        
        # Actually, B is a 5-subset. It contains current_smallest (4 pts).
        # So B = current_smallest + {x}
        # We need to check if the other 4 4-subsets of B are uncovered.
        
        # To maintain lexicographic order or just find any solution:
        # We iterate x from 0 to 10
        for x in points:
            if x in current_smallest:
                continue
            B = tuple(sorted(current_smallest + (x,)))
            
            # Check validity
            valid = True
            new_subs = []
            for sub in combinations(B, 4):
                if sub in covered:
                    valid = False
                    break
                new_subs.append(sub)
            
            if valid:
                # Add
                blocks.append(B)
                for sub in new_subs:
                    covered.add(sub)
                
                if backtrack():
                    return True
                
                # Undo
                blocks.pop()
                for sub in new_subs:
                    covered.remove(sub)
                    
        return False

    start_time = time.time()
    if backtrack():
        return blocks
    else:
        return []

def construct():
    # Get S(4,5,11) blocks
    s4511_blocks = _solve_s4511()
    
    if not s4511_blocks:
        # Fallback to greedy if solver fails (should not happen)
        # ... implement greedy ...
        return []
        
    result = []
    # Type 1: {11} U B
    for b in s4511_blocks:
        result.append(list(b) + [11])
        
    # Type 2: Complement of B in {0..10}
    full_set = set(range(11))
    for b in s4511_blocks:
        comp = list(full_set - set(b))
        result.append(comp)
        
    return result
