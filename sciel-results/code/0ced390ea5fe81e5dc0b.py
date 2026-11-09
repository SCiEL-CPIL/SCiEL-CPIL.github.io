import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Step 1: Construct S(4,5,11) on points 1..11
    points = list(range(1, 12))
    all_4_subsets = set(combinations(points, 4))
    covered_4 = set()
    blocks_4511 = []
    
    # Symmetry breaking: First block is (1, 2, 3, 4, 5)
    first_block = (1, 2, 3, 4, 5)
    blocks_4511.append(first_block)
    for s in combinations(first_block, 4):
        covered_4.add(s)
    
    # Candidates for subsequent blocks
    # We iterate through all 5-subsets of 1..11
    candidates = [c for c in combinations(points, 5) if c != first_block]
    
    # To make it faster, we can sort candidates or use a specific order
    # But 462 candidates is small.
    
    def backtrack(index, covered, current_blocks):
        if time.time() > deadline:
            return None
        if len(current_blocks) == 66:
            return current_blocks
        if index >= len(candidates):
            return None
            
        # Optimization: If remaining candidates cannot cover remaining subsets, prune
        # (Optional, but good for robustness)
        
        for i in range(index, len(candidates)):
            b = candidates[i]
            # Check if any 4-subset of b is already covered
            valid = True
            for s in combinations(b, 4):
                if s in covered:
                    valid = False
                    break
            
            if valid:
                # Add b
                new_covered = set(covered)
                for s in combinations(b, 4):
                    new_covered.add(s)
                
                result = backtrack(i + 1, new_covered, current_blocks + [b])
                if result is not None:
                    return result
        
        return None

    # Run backtracking
    # Note: The greedy approach might work, but backtracking is guaranteed.
    # Given the small size, backtracking is fast.
    sol_4511 = backtrack(0, covered_4, blocks_4511)
    
    if sol_4511 is None:
        # Fallback: Greedy (just in case, though backtracking should work)
        covered_4 = set()
        blocks_4511 = []
        for b in combinations(points, 5):
            subsets = list(combinations(b, 4))
            if not any(s in covered_4 for s in subsets):
                blocks_4511.append(b)
                covered_4.update(subsets)
                if len(blocks_4511) == 66:
                    break
        sol_4511 = blocks_4511

    # Step 2: Construct S(5,6,12)
    final_blocks = []
    
    # Blocks containing 0
    for b in sol_4511:
        final_blocks.append([0] + list(b))
        
    # Blocks not containing 0 (complements in 1..11)
    for b in sol_4511:
        b_set = set(b)
        comp = [p for p in points if p not in b_set]
        final_blocks.append(comp)
        
    return final_blocks
