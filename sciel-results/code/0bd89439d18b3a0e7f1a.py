import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    # Map block to its 5-subsets
    block_subsets = {}
    for b in all_blocks:
        block_subsets[b] = set(combinations(b, 5))
    
    # Map 5-subset to list of blocks containing it
    subset_to_blocks = {}
    for b in all_blocks:
        for s in block_subsets[b]:
            if s not in subset_to_blocks:
                subset_to_blocks[s] = []
            subset_to_blocks[s].append(b)
            
    best_blocks = []
    
    # Symmetry breaking: Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_subsets = block_subsets[first_block]
    
    # Candidates for second block
    # Must intersect first block in 0, 1, 2, or 3 points.
    # And must not share any 5-subsets with first block (which is true if intersection < 5).
    # Actually, if they share 5 points, they share a 5-subset.
    # So intersection must be <= 4?
    # In S(5,6,12), max intersection is 3?
    # Let's check: If two blocks intersect in 4 points, they share 4 choose 4 = 1 4-subset.
    # Do they share a 5-subset? No.
    # But in S(5,6,12), the intersection sizes are 0, 1, 2, 3.
    # So we can restrict second block to intersect first block in 0, 1, 2, or 3 points.
    
    second_block_candidates = []
    for b in all_blocks:
        if b == first_block:
            continue
        inter = len(set(b).intersection(set(first_block)))
        if inter <= 3:
            second_block_candidates.append(b)
            
    def search(current_blocks, covered, depth):
        nonlocal best_blocks
        
        if time.time() - start_time > timeout:
            return
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = list(current_blocks)
            
        if len(current_blocks) == 132:
            return True
        
        # MRV: Find the 5-subset with fewest available blocks
        min_count = float('inf')
        target_subset = None
        target_blocks = []
        
        # Iterate through all 5-subsets not in covered
        # There are 792 5-subsets.
        # Checking all of them every step is expensive (792 * 132 * cost).
        # Optimization: Only check subsets of the *last added block*?
        # No, we need global MRV.
        # However, we can optimize by only considering subsets that are "active".
        # A subset is active if it is not covered.
        
        # To speed up, we can pre-compute the list of all 5-subsets.
        all_5_subsets = list(combinations(range(12), 5))
        
        # This loop is O(792).
        for s in all_5_subsets:
            if s in covered:
                continue
            # Count available blocks containing s
            count = 0
            avail = []
            for b in subset_to_blocks[s]:
                # Check if b is available (all its 5-subsets are not in covered)
                # This check is expensive if done naively.
                # We can maintain a "conflict" count or just check.
                # Since depth is small, and we prune, let's try.
                # Optimization: If any 5-subset of b is in covered, b is not available.
                is_avail = True
                for sub in block_subsets[b]:
                    if sub in covered:
                        is_avail = False
                        break
                if is_avail:
                    count += 1
                    avail.append(b)
            
            if count < min_count:
                min_count = count
                target_subset = s
                target_blocks = avail
                if count == 0:
                    break # Dead end
                
        if target_subset is None:
            # All covered? Should not happen before 132 blocks.
            return False
            
        if min_count == 0:
            return False
            
        # Try each block in target_blocks
        for b in target_blocks:
            new_covered = covered.union(block_subsets[b])
            if search(current_blocks + [b], new_covered, depth + 1):
                return True
                
        return False
        
    # Initial call
    # We fix the first block to break symmetry
    initial_covered = set(first_block_subsets)
    search([first_block], initial_covered, 1)
    
    # If we didn't find 132, we might have found a good packing.
    # But we should probably try without fixing the first block if it fails?
    # No, symmetry breaking is safe for finding *a* solution of size 132.
    # If a solution of size 132 exists, one exists with first block (0,1,2,3,4,5).
    
    if len(best_blocks) < 132:
        # Fallback: Greedy? Or just return what we have.
        # Given the constraints, the backtracking should find it.
        pass
        
    return [list(b) for b in best_blocks]
