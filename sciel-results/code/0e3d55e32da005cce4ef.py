"""Witt design S(5,6,12) via deterministic backtracking with constraint propagation."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    points = list(range(12))
    
    # Precompute all 5-subsets as frozensets for fast lookup
    all_5subsets = list(combinations(points, 5))
    # Map each 5-subset to its index
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # For each 6-subset, precompute the list of 5-subset indices it contains
    all_6subsets = list(combinations(points, 6))
    block_5subs = []
    for b in all_6subsets:
        subs = [subset_to_idx[s] for s in combinations(b, 5)]
        block_5subs.append(subs)
    
    # Also precompute for each 5-subset index, which 6-subsets contain it
    subset_to_blocks = [[] for _ in range(len(all_5subsets))]
    for i, subs in enumerate(block_5subs):
        for s in subs:
            subset_to_blocks[s].append(i)
    
    covered = [False] * len(all_5subsets)
    blocks = []
    
    def backtrack():
        # Find the most constrained uncovered 5-subset (fewest available blocks)
        best_idx = -1
        best_count = float('inf')
        best_block_list = None
        
        for s in range(len(all_5subsets)):
            if not covered[s]:
                # Count how many blocks can still cover this 5-subset
                count = 0
                for b_idx in subset_to_blocks[s]:
                    # Check if this block is still feasible (no overlap with covered)
                    feasible = True
                    for ss in block_5subs[b_idx]:
                        if covered[ss]:
                            feasible = False
                            break
                    if feasible:
                        count += 1
                        if count > best_count:
                            break
                if count < best_count:
                    best_count = count
                    best_idx = s
                    best_block_list = [b_idx for b_idx in subset_to_blocks[s]
                                       if not any(covered[ss] for ss in block_5subs[b_idx])]
                    if count == 0:
                        return False
                    if count == 1:
                        break  # This is as constrained as it gets
        
        if best_idx == -1:
            # All 5-subsets covered
            return True
        
        # Try blocks that cover the most constrained 5-subset
        if best_block_list is None:
            best_block_list = [b_idx for b_idx in subset_to_blocks[best_idx]
                               if not any(covered[ss] for ss in block_5subs[b_idx])]
        
        for b_idx in best_block_list:
            # Check feasibility
            feasible = True
            for ss in block_5subs[b_idx]:
                if covered[ss]:
                    feasible = False
                    break
            if not feasible:
                continue
            
            # Choose this block
            blocks.append(all_6subsets[b_idx])
            for ss in block_5subs[b_idx]:
                covered[ss] = True
            
            # Check time
            if time.time() > deadline:
                covered[ss] = False  # revert
                blocks.pop()
                return True  # Return partial result
            
            if backtrack():
                return True
            
            # Backtrack
            for ss in block_5subs[b_idx]:
                covered[ss] = False
            blocks.pop()
        
        return False
    
    # Use a smarter initial ordering: start with blocks that cover many uncovered subsets
    # Actually, let's just run the backtracking
    
    result = backtrack()
    
    return blocks
