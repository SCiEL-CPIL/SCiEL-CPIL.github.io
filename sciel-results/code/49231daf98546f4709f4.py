import time
from itertools import combinations

def construct():
    # Deadline guard - use more of the available 150s
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = range(12)
    
    # Precompute all 5-subsets and 6-subsets
    all_5_subsets = list(combinations(points, 5))
    all_6_subsets = list(combinations(points, 6))
    
    # Index for 5-subsets
    five_idx = {s: i for i, s in enumerate(all_5_subsets)}
    n5 = len(all_5_subsets)  # 792
    
    # For each 6-subset, precompute its 5-subset indices
    block_to_5idx = {}
    for b in all_6_subsets:
        block_to_5idx[b] = [five_idx[s] for s in combinations(b, 5)]
    
    # For each 5-subset index, precompute which 6-subsets contain it
    five_to_sixes = [[] for _ in range(n5)]
    for b in all_6_subsets:
        for idx in block_to_5idx[b]:
            five_to_sixes[idx].append(b)
    
    # Use a boolean array for covered 5-subsets (faster than set)
    covered = [False] * n5
    covered_count = 0
    blocks = []
    
    # Symmetry breaking:
    # The first block must contain {0,1,2,3,4}. 
    # By symmetry, we can assume the 6th element is 5.
    # So Block 0 is {0,1,2,3,4,5}.
    
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_5idx = block_to_5idx[first_block]
    
    blocks.append(first_block)
    for idx in first_block_5idx:
        covered[idx] = True
    covered_count = len(first_block_5idx)
    
    # Maintain a list of uncovered 5-subset indices for fast lookup
    uncovered = [i for i in range(n5) if not covered[i]]
    
    # Now we need 131 more blocks.
    # We will search for the next block.
    
    def backtrack():
        nonlocal covered_count, uncovered
        
        if len(blocks) == 132:
            return True
        
        if time.time() > deadline:
            return False
            
        if not uncovered:
            return False
            
        # Take the first uncovered 5-subset (lexicographic order)
        target_idx = uncovered[0]
        target_5 = all_5_subsets[target_idx]
        
        # Find all 6-subsets that contain target_5
        candidates = five_to_sixes[target_idx]
        
        # Filter candidates: no overlap in 5-subsets
        valid_candidates = []
        for b in candidates:
            # Check if this block is valid (no overlap in 5-subsets)
            block_5idx = block_to_5idx[b]
            valid = True
            for idx in block_5idx:
                if covered[idx]:
                    valid = False
                    break
            if valid:
                valid_candidates.append(b)
                
        # Try each candidate - order by how many new 5-subsets it covers (most constrained first)
        # Actually for exact cover, we want to try candidates that cover the most uncovered subsets
        valid_candidates.sort(key=lambda b: sum(1 for idx in block_to_5idx[b] if not covered[idx]), reverse=True)
        
        for b in valid_candidates:
            # Add block
            blocks.append(b)
            block_5idx = block_to_5idx[b]
            for idx in block_5idx:
                covered[idx] = True
            covered_count += len(block_5idx)
            
            # Update uncovered list
            uncovered = [i for i in uncovered if not covered[i]]
            
            if backtrack():
                return True
                
            # Undo
            blocks.pop()
            for idx in block_5idx:
                covered[idx] = False
            covered_count -= len(block_5idx)
            
            # Rebuild uncovered list - we lost some, so rebuild from scratch
            uncovered = [i for i in range(n5) if not covered[i]]
            
        return False
        
    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback: Return what we have
        return [list(b) for b in blocks]
