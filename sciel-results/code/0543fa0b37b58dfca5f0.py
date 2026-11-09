import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    points = range(12)
    all_6_subsets = list(combinations(points, 6))
    
    # Precompute 5-subsets for each 6-subset
    subset_map = {}
    for b in all_6_subsets:
        subset_map[b] = list(combinations(b, 5))
        
    covered = set()
    blocks = []
    
    # Backtracking
    # To speed up, we can sort the 6-subsets by some heuristic?
    # Lexicographic order is fine.
    
    def backtrack(index):
        if time.time() > deadline:
            return False
        
        if index == len(all_6_subsets):
            return True
        
        b = all_6_subsets[index]
        subsets = subset_map[b]
        
        # Check conflict
        conflict = False
        for s in subsets:
            if s in covered:
                conflict = True
                break
                
        if conflict:
            return backtrack(index + 1)
        
        # Add block
        blocks.append(b)
        for s in subsets:
            covered.add(s)
            
        if backtrack(index + 1):
            return True
            
        # Backtrack
        blocks.pop()
        for s in subsets:
            covered.remove(s)
            
        return False

    backtrack(0)
    return blocks
