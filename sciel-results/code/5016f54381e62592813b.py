import time
from itertools import combinations

# Pre-computation
def _init():
    points = range(12)
    blocks = list(combinations(points, 6))
    block_5s = [list(combinations(b, 5)) for b in blocks]
    
    sub5_to_idx = {}
    idx = 0
    for s in combinations(points, 5):
        sub5_to_idx[s] = idx
        idx += 1
        
    num_5s = idx
    
    block_masks = [0] * len(blocks)
    blocks_containing = [[] for _ in range(num_5s)]
    
    for i, b in enumerate(blocks):
        mask = 0
        for s in block_5s[i]:
            s_idx = sub5_to_idx[s]
            mask |= (1 << s_idx)
            blocks_containing[s_idx].append(i)
        block_masks[i] = mask
        
    return blocks, block_masks, blocks_containing

_BLOCKS, _BLOCK_MASKS, _BLOCKS_CONTAINING = _init()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = []
    path = []
    
    # Symmetry breaking: Fix first block to be (0, 1, 2, 3, 4, 5)
    start_block_idx = 0
    path.append(start_block_idx)
    best_blocks = [start_block_idx]
    
    initial_mask = _BLOCK_MASKS[start_block_idx]
    
    def dfs(used_mask, depth):
        nonlocal best_blocks, path, deadline
        
        if time.time() > deadline:
            return False
            
        if depth > len(best_blocks):
            best_blocks = path.copy()
            
        if depth == 132:
            return True

        min_count = 100
        best_5_idx = -1
        candidates = []
        
        for s_idx in range(792):
            if used_mask & (1 << s_idx):
                continue
            
            count = 0
            cand_list = []
            for b_idx in _BLOCKS_CONTAINING[s_idx]:
                if _BLOCK_MASKS[b_idx] & used_mask:
                    continue
                count += 1
                cand_list.append(b_idx)
            
            if count == 0:
                return False
            
            if count < min_count:
                min_count = count
                best_5_idx = s_idx
                candidates = cand_list
                if min_count == 1:
                    break

        if best_5_idx == -1:
            return False

        for b_idx in candidates:
            path.append(b_idx)
            if dfs(used_mask | _BLOCK_MASKS[b_idx], depth + 1):
                return True
            path.pop()
            
        return False

    dfs(initial_mask, 1)
    
    return [list(_BLOCKS[i]) for i in best_blocks]
