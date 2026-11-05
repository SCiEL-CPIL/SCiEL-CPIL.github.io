import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Fast Exact Cover Solver (Algorithm X with MRV heuristic)
    # The problem is equivalent to finding an exact cover of the 792 5-subsets
    # using the 924 possible 6-subsets. The optimal solution (Witt design)
    # covers every 5-subset exactly once.
    
    points = range(12)
    all_5_subs = list(combinations(points, 5))
    n_5 = len(all_5_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5_subs)}
    
    all_6_blocks = list(combinations(points, 6))
    n_6 = len(all_6_blocks)
    
    # Pre-compute the 5-subsets for each 6-block
    block_to_subs = []
    for b in all_6_blocks:
        subs = [sub_to_idx[s] for s in combinations(b, 5)]
        block_to_subs.append(subs)
        
    # For each 5-subset, list the blocks that cover it
    sub_to_blocks = [[] for _ in range(n_5)]
    for j, subs in enumerate(block_to_subs):
        for i in subs:
            sub_to_blocks[i].append(j)
            
    solution = []
    covered = [False] * n_5
    used_blocks = [False] * n_6
    
    def solve():
        # Find the uncovered 5-subset with the minimum number of available blocks (MRV)
        min_count = float('inf')
        min_sub = -1
        for i in range(n_5):
            if not covered[i]:
                count = 0
                for j in sub_to_blocks[i]:
                    if not used_blocks[j]:
                        count += 1
                        if count < min_count:
                            min_count = count
                            min_sub = i
                            if min_count == 0:
                                return False
                if count == 0:
                    return False
        if min_sub == -1:
            return True # All covered
            
        # Try all blocks that cover min_sub
        for j in sub_to_blocks[min_sub]:
            if used_blocks[j]:
                continue
            # Check if this block conflicts with already covered subsets
            valid = True
            for i in block_to_subs[j]:
                if covered[i]:
                    valid = False
                    break
            if not valid:
                continue
                
            # Apply choice
            solution.append(j)
            used_blocks[j] = True
            for i in block_to_subs[j]:
                covered[i] = True
                
            if solve():
                return True
                
            # Backtrack
            solution.pop()
            used_blocks[j] = False
            for i in block_to_subs[j]:
                covered[i] = False
                
        return False

    if solve():
        return [list(all_6_blocks[j]) for j in solution]
        
    # Fallback: Greedy
    covered_set = set()
    greedy_blocks = []
    for b in all_6_blocks:
        subs = list(combinations(b, 5))
        if not any(s in covered_set for s in subs):
            greedy_blocks.append(list(b))
            covered_set.update(subs)
    return greedy_blocks
