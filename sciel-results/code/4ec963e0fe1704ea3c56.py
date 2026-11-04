import numpy as np
import itertools
import time
import math

def construct(v, k, t):
    # Fallback for non-standard parameters
    if v != 19 or k != 8 or t != 5:
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

    start_time = time.time()
    deadline = 145.0

    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)

    # Precompute indices of t-subsets covered by each k-block
    c_k_t = math.comb(k, t)
    all_indices = np.empty((n_k_blocks, c_k_t), dtype=np.int32)
    for b_idx, block in enumerate(k_blocks):
        idxs = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        all_indices[b_idx] = idxs

    best_block_indices = None
    best_count = float('inf')

    # Run multiple restarts
    # 100 restarts should be well within time limit
    for restart in range(100):
        if time.time() > deadline:
            break

        uncovered = np.ones(n_t_subsets, dtype=bool)
        current_block_indices = []

        while np.any(uncovered):
            # Vectorized gain calculation
            counts = np.sum(uncovered[all_indices], axis=1)
            
            # Add noise to break ties
            noise = np.random.normal(0, 0.5, size=n_k_blocks)
            counts += noise
            
            best_b = np.argmax(counts)
            current_block_indices.append(best_b)
            uncovered[all_indices[best_b]] = False

        if len(current_block_indices) < best_count:
            best_count = len(current_block_indices)
            best_block_indices = current_block_indices.copy()

    if best_block_indices is None:
        return []

    result = [list(k_blocks[i]) for i in best_block_indices]
    return result
