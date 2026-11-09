import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Algebraic construction of S(5,6,12) from Golay Code
    # 1. Generator Matrix
    P = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.uint8)
    
    I12 = np.eye(12, dtype=np.uint8)
    G = np.hstack((I12, P))
    
    # 2. Generate all codewords
    masks = np.array([list((i >> j) & 1 for j in range(12)) for i in range(4096)], dtype=np.uint8)
    all_codewords = (masks @ G) % 2
    weights = np.sum(all_codewords, axis=1)
    
    # 3. Find a dodecad
    dodecad_indices = np.where(weights == 12)[0]
    if len(dodecad_indices) == 0:
        return _fallback_greedy()
        
    dodecad_mask = all_codewords[dodecad_indices[0]]
    dodecad_points = np.where(dodecad_mask == 1)[0]
    
    # 4. Extract blocks
    blocks = []
    for i in range(4096):
        if weights[i] == 6:
            c = all_codewords[i]
            if not np.any(c & (1 - dodecad_mask)):
                block_indices = np.where(c == 1)[0]
                mapped_block = [int(np.where(dodecad_points == idx)[0][0]) for idx in block_indices]
                blocks.append(tuple(sorted(mapped_block)))
                
    if len(blocks) == 132:
        return [list(b) for b in blocks]
        
    return _fallback_greedy()

def _fallback_greedy():
    # Simple greedy fallback
    from itertools import combinations
    points = list(range(12))
    candidates = list(combinations(points, 6))
    covered = set()
    blocks = []
    for b in candidates:
        b_set = set(b)
        # Check 5-subsets
        valid = True
        for sub in combinations(b, 5):
            if frozenset(sub) in covered:
                valid = False
                break
        if valid:
            blocks.append(list(b))
            for sub in combinations(b, 5):
                covered.add(frozenset(sub))
            if len(blocks) == 132:
                break
    return blocks
