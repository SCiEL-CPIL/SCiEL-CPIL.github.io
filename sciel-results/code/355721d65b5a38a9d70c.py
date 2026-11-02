import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1

    # 2. Greedy Search with Restart and Polish
    best_blocks = None
    best_len = float('inf')

    # We will run multiple restarts
    # To diversify, we can randomize the order of tie-breaking
    while time.time() < deadline:
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            # Calculate gain for all blocks
            gains = uncovered[BlockTsets].sum(axis=1)

            max_gain = gains.max()
            if max_gain == 0:
                break

            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]

            # Randomly select one to diversify
            best_block_idx = candidates[np.random.randint(len(candidates))]

            blocks.append(best_block_idx)

            # Update uncovered
            uncovered[BlockTsets[best_block_idx]] = False

        # Polish: Remove redundant blocks
        if len(blocks) < best_len:
            cover_count = np.zeros(num_tsets, dtype=np.int32)
            for b in blocks:
                cover_count[BlockTsets[b]] += 1

            # Iterate from end to start
            i = len(blocks) - 1
            while i >= 0:
                b = blocks[i]
                can_remove = True
                for ts in BlockTsets[b]:
                    if cover_count[ts] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in BlockTsets[b]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                else:
                    i -= 1

            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]

    # Convert indices back to actual blocks (lists of points)
    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
