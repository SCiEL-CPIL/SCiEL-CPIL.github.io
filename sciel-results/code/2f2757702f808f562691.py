import time
from itertools import combinations

# Global variables for speed
ALL_BLOCKS = []
BLOCK_SUBSETS = {}
START_TIME = 0
TIME_LIMIT = 110.0  # Leave margin for 120s limit

def construct():
    global ALL_BLOCKS, BLOCK_SUBSETS, START_TIME
    
    START_TIME = time.time()
    
    # 1. Precompute 5-subsets as bitmasks
    # A 5-subset is a tuple of 5 distinct ints from 0..11
    # Mask: 12 bits
    def get_mask(subset):
        mask = 0
        for x in subset:
            mask |= (1 << x)
        return mask

    # 2. Generate all 6-subsets and their 5-subsets
    points = range(12)
    all_6_subsets = list(combinations(points, 6))
    
    # We want to enforce symmetry breaking: First block is (0, 1, 2, 3, 4, 5)
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    # Precompute subsets for all blocks
    for b in all_6_subsets:
        subsets = list(combinations(b, 5))
        masks = [get_mask(s) for s in subsets]
        BLOCK_SUBSETS[b] = masks

    # Initialize state
    covered = set(BLOCK_SUBSETS[first_block])
    blocks = [list(first_block)]
    
    # Recursive search
    def backtrack(start_idx, covered, blocks):
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            return False
        
        if len(blocks) == 132:
            return True
            
        # Heuristic: Try blocks that appear early in the sorted list
        # This helps find a valid packing quickly if one exists
        for i in range(start_idx, len(ALL_BLOCKS)):
            b = ALL_BLOCKS[i]
            masks = BLOCK_SUBSETS[b]
            
            # Check if any 5-subset is already covered
            conflict = False
            for m in masks:
                if m in covered:
                    conflict = True
                    break
            
            if not conflict:
                # Add block
                covered.update(masks)
                blocks.append(list(b))
                
                # Recurse
                if backtrack(i + 1, covered, blocks):
                    return True
                
                # Undo
                blocks.pop()
                for m in masks:
                    covered.discard(m)
                    
                # Time check again
                if time.time() - START_TIME > TIME_LIMIT:
                    return False

        return False

    # Run search
    # We start searching for the 2nd block from index first_block_idx + 1
    backtrack(first_block_idx + 1, covered, blocks)
    
    return blocks

# Pre-compute ALL_BLOCKS globally to avoid re-computation if called multiple times (though construct is usually called once)
# But we need to ensure ALL_BLOCKS is populated before backtrack uses it.
# The current code populates ALL_BLOCKS inside construct. Let's move it out or ensure it's done.
# Actually, the code above defines ALL_BLOCKS inside construct but uses it in backtrack.
# Wait, ALL_BLOCKS is global but assigned inside construct.
# Let's refine: populate ALL_BLOCKS at module level or inside construct before defining backtrack.

# Revised plan:
# Populate ALL_BLOCKS at the start of construct.

def construct():
    global ALL_BLOCKS, BLOCK_SUBSETS, START_TIME
    
    START_TIME = time.time()
    
    points = range(12)
    all_6_subsets = list(combinations(points, 6))
    ALL_BLOCKS = all_6_subsets
    
    def get_mask(subset):
        mask = 0
        for x in subset:
            mask |= (1 << x)
        return mask

    BLOCK_SUBSETS = {}
    for b in all_6_subsets:
        subsets = list(combinations(b, 5))
        masks = [get_mask(s) for s in subsets]
        BLOCK_SUBSETS[b] = masks

    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6_subsets.index(first_block)
    
    covered = set(BLOCK_SUBSETS[first_block])
    blocks = [list(first_block)]
    
    def backtrack(start_idx, covered, blocks):
        if time.time() - START_TIME > TIME_LIMIT:
            return False
        
        if len(blocks) == 132:
            return True
            
        for i in range(start_idx, len(ALL_BLOCKS)):
            b = ALL_BLOCKS[i]
            masks = BLOCK_SUBSETS[b]
            
            # Fast check
            if any(m in covered for m in masks):
                continue
            
            covered.update(masks)
            blocks.append(list(b))
            
            if backtrack(i + 1, covered, blocks):
                return True
            
            blocks.pop()
            for m in masks:
                covered.discard(m)
                
            if time.time() - START_TIME > TIME_LIMIT:
                return False

        return False

    backtrack(first_block_idx + 1, covered, blocks)
    
    return blocks
