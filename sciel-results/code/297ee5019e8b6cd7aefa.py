import itertools
import time
import numpy as np
import random
import sys

def construct(v, k, t):
    # Precomputation
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {ts: i for i, ts in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # 2. Generate all k-blocks and their masks/indices
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    masks = [0] * n_k_blocks
    indices = [[] for _ in range(n_k_blocks)]
    block_points = [list(b) for b in k_blocks]
    
    for i, block in enumerate(k_blocks):
        # Find all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            idx = t_subsets_set[t_sub]
            indices[i].append(idx)
            masks[i] |= (1 << idx)
            
    # Local Search
    deadline = time.time() + 110
    best_blocks = None
    best_size = float('inf')
    
    # State
    current_blocks = []
    counts = np.zeros(n_t_subsets, dtype=np.int32)
    uncovered_mask = (1 << n_t_subsets) - 1
    current_size = 0
    stuck_counter = 0
    
    while time.time() < deadline:
        # If not covered, Greedy Step
        if uncovered_mask != 0:
            # Find best block
            best_gain = -1
            best_b = -1
            # Optimization: Only check blocks that have potential gain?
            # With bitsets, just scan all.
            for b in range(n_k_blocks):
                gain = (masks[b] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
                # Early break if gain is max possible? Max is k choose t = 56.
                # if gain == 56: break # Unlikely to be max unless uncovered is small subset of one block
            
            if best_b != -1:
                current_blocks.append(best_b)
                counts[indices[best_b]] += 1
                uncovered_mask &= ~masks[best_b]
                current_size += 1
                stuck_counter = 0
            continue
        
        # Covered: Reduction
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid bias
            random.shuffle(current_blocks)
            for i in range(len(current_blocks)):
                b = current_blocks[i]
                counts[indices[b]] -= 1
                if np.any(counts == 0):
                    counts[indices[b]] += 1
                else:
                    current_blocks.pop(i)
                    current_size -= 1
                    improved = True
                    break
        
        if current_size < best_size:
            best_size = current_size
            best_blocks = current_blocks.copy()
            stuck_counter = 0
        else:
            stuck_counter += 1
        
        # Perturbation
        if time.time() > deadline - 5:
            break
            
        if stuck_counter > 10 or current_size == 0:
            # Restart
            current_blocks = []
            counts = np.zeros(n_t_subsets, dtype=np.int32)
            uncovered_mask = (1 << n_t_subsets) - 1
            current_size = 0
            stuck_counter = 0
        else:
            # Swap
            if len(current_blocks) > 0:
                b_out = random.choice(current_blocks)
                # Pick b_in
                # To be efficient, pick from a random sample of non-current blocks
                # or just pick random until we find one not in current
                # Since current is small (~300) vs total (75k), random pick is likely not in current
                b_in = random.randint(0, n_k_blocks - 1)
                if b_in in current_blocks:
                    # Retry a few times or just skip
                    pass
                else:
                    # Tentative
                    counts[indices[b_out]] -= 1
                    counts[indices[b_in]] += 1
                    if np.any(counts == 0):
                        # Revert
                        counts[indices[b_out]] += 1
                        counts[indices[b_in]] -= 1
                    else:
                        # Accept
                        current_blocks.remove(b_out)
                        current_blocks.append(b_in)

    # Convert best_blocks indices to point lists
    if best_blocks is None:
        return []
    return [block_points[i] for i in best_blocks]
