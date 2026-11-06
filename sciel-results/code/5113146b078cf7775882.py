# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Generate all 924 blocks
    all_blocks = list(combinations(range(12), 6))
    n_blocks = len(all_blocks)
    
    # Map 5-subsets to indices
    all_5sets = list(combinations(range(12), 5))
    idx_5set = {s: i for i, s in enumerate(all_5sets)}
    n_5sets = len(all_5sets)
    
    # Precompute 5-subset indices for each block
    block_5sets_idx = []
    for b in all_blocks:
        indices = [idx_5set[s] for s in combinations(b, 5)]
        block_5sets_idx.append(indices)
        
    # State for backtracking
    # We enforce lexicographic order of block indices to ensure we find the unique solution path
    # without branching into isomorphic copies.
    
    used = [False] * n_5sets
    current_blocks = []
    
    def backtrack(last_idx):
        if time.time() - start_time > timeout:
            return False
            
        if len(current_blocks) == 132:
            return True
            
        # Find the next valid block with index > last_idx
        for i in range(last_idx + 1, n_blocks):
            b_indices = block_5sets_idx[i]
            
            # Check if any 5-subset is already used
            conflict = False
            for idx in b_indices:
                if used[idx]:
                    conflict = True
                    break
                    
            if conflict:
                continue
                
            # Accept block i
            current_blocks.append(all_blocks[i])
            for idx in b_indices:
                used[idx] = True
                
            if backtrack(i):
                return True
                
            # Undo
            for idx in b_indices:
                used[idx] = False
            current_blocks.pop()
            
        return False

    # Start with the first block in lex order: (0, 1, 2, 3, 4, 5)
    # Its index is 0.
    first_block_idx = 0
    current_blocks.append(all_blocks[first_block_idx])
    for idx in block_5sets_idx[first_block_idx]:
        used[idx] = True
        
    if backtrack(first_block_idx):
        return [list(b) for b in current_blocks]
        
    # Fallback (should not happen)
    return [list(b) for b in current_blocks]

# EVOLVE-BLOCK-END
