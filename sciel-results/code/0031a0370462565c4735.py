import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    # Parameters for the specific task
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy if parameters are different
        # But the task is specific to C(19,7,4)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    # Precomputation for C(19, 7, 4)
    points = range(19)
    subsets_4 = list(itertools.combinations(points, 4))
    n_4 = len(subsets_4)
    subsets_7 = list(itertools.combinations(points, 7))
    n_7 = len(subsets_7)
    
    # Map 4-subsets to indices
    subset_4_index = {s: i for i, s in enumerate(subsets_4)}
    
    # Precompute for each 7-block, which 4-subsets it covers
    # And for each 4-subset, which 7-blocks cover it
    block_indices = [] # List of lists of 4-indices
    candidates_for_4 = [[] for _ in range(n_4)]
    
    for b_idx, block in enumerate(subsets_7):
        block_set = set(block)
        covered_4s = []
        for s in subsets_4:
            if set(s).issubset(block_set):
                covered_4s.append(subset_4_index[s])
                candidates_for_4[subset_4_index[s]].append(b_idx)
        block_indices.append(covered_4s)
        
    # Convert to numpy arrays for faster operations
    # block_indices_np: shape (n_7, 35)
    block_indices_np = np.array(block_indices, dtype=np.int32)
    
    start_time = time.time()
    deadline = start_time + 140.0
    best_blocks = None
    best_len = float('inf')
    
    while time.time() < deadline:
        uncovered = np.ones(n_4, dtype=bool)
        coverage_count = np.zeros(n_4, dtype=np.int32)
        blocks = []
        uncovered_indices = set(range(n_4))
        
        while uncovered_indices:
            # Pick a random uncovered 4-subset
            s_idx = random.choice(tuple(uncovered_indices))
            
            # Get candidate blocks covering s_idx
            cands = candidates_for_4[s_idx]
            
            best_b_idx = -1
            best_score = -1
            
            # Shuffle candidates to break ties randomly
            random.shuffle(cands)
            
            for b_idx in cands:
                # Count how many uncovered 4-subsets this block covers
                score = np.sum(uncovered[block_indices_np[b_idx]])
                if score > best_score:
                    best_score = score
                    best_b_idx = b_idx
            
            # Add the best block
            blocks.append(best_b_idx)
            
            # Update uncovered and coverage_count
            idxs = block_indices_np[best_b_idx]
            for idx in idxs:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_indices.remove(idx)
                coverage_count[idx] += 1
        
        # Deletion Phase
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow popping
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                essential = False
                idxs = block_indices_np[b_idx]
                for idx in idxs:
                    if coverage_count[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    for idx in idxs:
                        coverage_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
    # Convert block indices back to lists of points
    result = [list(subsets_7[b_idx]) for b_idx in best_blocks]
    return result
