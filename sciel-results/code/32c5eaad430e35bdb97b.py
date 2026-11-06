import sys
from itertools import combinations

sys.setrecursionlimit(200)

def construct():
    all_points = range(12)
    all_blocks = list(combinations(all_points, 6))
    
    # Pre-compute 5-subsets for each block to speed up conflict checking
    block_5_subsets = {}
    for b in all_blocks:
        block_5_subsets[b] = list(combinations(b, 5))
        
    # Map 5-subset to index for faster set operations? 
    # Or just use a set of 5-tuples.
    
    first_block = (0, 1, 2, 3, 4, 5)
    covered = set(block_5_subsets[first_block])
    blocks = [first_block]
    
    def backtrack():
        if len(blocks) == 132:
            return True
        
        # Pruning
        remaining_blocks_needed = 132 - len(blocks)
        remaining_5_subsets = 792 - len(covered)
        if remaining_5_subsets < remaining_blocks_needed * 6:
            return False
            
        # Try adding each block
        for b in all_blocks:
            # Check if b conflicts
            # If b is already in blocks, skip? 
            # We don't track "in blocks" explicitly in a set, 
            # but since we only add valid blocks, and we don't remove them from the 
            # "all_blocks" iteration, we might try to add a block that is already there?
            # No, we iterate over all_blocks. If b is already in blocks, 
            # its 5-subsets are in covered, so it will be rejected by the conflict check.
            
            valid = True
            for s in block_5_subsets[b]:
                if s in covered:
                    valid = False
                    break
            
            if valid:
                # Add b
                blocks.append(b)
                added_subsets = block_5_subsets[b]
                covered.update(added_subsets)
                
                if backtrack():
                    return True
                
                # Backtrack
                blocks.pop()
                for s in added_subsets:
                    covered.discard(s)
                    
        return False

    if backtrack():
        return [list(b) for b in blocks]
    else:
        # Fallback
        covered, blocks = set(), []
        for b in all_blocks:
            subsets = block_5_subsets[b]
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
