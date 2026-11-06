import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = range(12)
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    five_idx = {s: i for i, s in enumerate(all_5_subsets)}
    n5 = len(all_5_subsets)  # 792
    
    # Precompute 5-subset indices for each 6-subset
    # Store as tuple for faster iteration?
    block_to_5idx = {}
    for b in all_6_subsets:
        block_to_5idx[b] = tuple(five_idx[s] for s in combinations(b, 5))
        
    # Precompute which 6-subsets contain a given 5-subset
    five_to_sixes = [[] for _ in range(n5)]
    for b in all_6_subsets:
        for idx in block_to_5idx[b]:
            five_to_sixes[idx].append(b)
            
    covered = [False] * n5
    blocks = []
    
    # Symmetry breaking: First block is (0,1,2,3,4,5)
    first_block = (0, 1, 2, 3, 4, 5)
    blocks.append(first_block)
    for idx in block_to_5idx[first_block]:
        covered[idx] = True
        
    # Count covered
    covered_count = 6
    
    def backtrack():
        nonlocal covered_count
        
        if len(blocks) == 132:
            return True
            
        if time.time() > deadline:
            return False
            
        # Check if remaining uncovered is multiple of 6
        remaining = n5 - covered_count
        if remaining % 6 != 0:
            return False
            
        # Find first uncovered 5-subset
        target_idx = -1
        for i in range(n5):
            if not covered[i]:
                target_idx = i
                break
                
        if target_idx == -1:
            return len(blocks) == 132
            
        # Get candidates
        candidates = five_to_sixes[target_idx]
        
        valid_candidates = []
        for b in candidates:
            # Check validity
            valid = True
            for idx in block_to_5idx[b]:
                if covered[idx]:
                    valid = False
                    break
            if valid:
                valid_candidates.append(b)
                
        # Heuristic: Try blocks that cover the most "constrained" points?
        # Or just randomize?
        # Standard: Most constrained variable (target_idx) is already chosen.
        # Value ordering: Try blocks that appear in fewer other potential solutions?
        # Simple: Sort by lexicographic order or random.
        # Let's try sorting by the number of common points with existing blocks?
        # Actually, for exact cover, any valid block is fine.
        # To improve speed, we can try blocks that are "more unique" or something.
        # Let's just iterate.
        
        for b in valid_candidates:
            blocks.append(b)
            for idx in block_to_5idx[b]:
                covered[idx] = True
            covered_count += 6
            
            if backtrack():
                return True
                
            blocks.pop()
            for idx in block_to_5idx[b]:
                covered[idx] = False
            covered_count -= 6
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        return [list(b) for b in blocks]
