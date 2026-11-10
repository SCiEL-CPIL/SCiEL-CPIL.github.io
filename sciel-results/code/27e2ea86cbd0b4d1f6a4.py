import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0  # Safety margin before 120s kill

    # Precompute t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute all candidate blocks (all k-subsets of v points)
    # For v=24, k=7: C(24,7) = 346104. This is the full candidate set.
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute coverage mask for each block as a Python int (bitset over n_tsubs)
    # Each bit i is 1 if t-subset i is contained in the block.
    block_masks = [0] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << tsub_index[combo])
        block_masks[bi] = mask

    # For each t-subset, precompute list of block indices that cover it
    # This is expensive: for each of C(24,3)=2024 t-subsets, find all blocks containing it.
    # A block contains a t-subset if all t elements are in the block.
    # We can build this by iterating blocks and setting membership.
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(all_blocks):
        block_set = set(block)
        for combo in itertools.combinations(block, t):
            tsub_to_blocks[tsub_index[combo]].append(bi)

    best_blocks = []
    best_count = float('inf')

    def local_search(block_indices, time_limit):
        """Remove redundant blocks using coverage counts."""
        blocks = list(block_indices)
        if not blocks:
            return blocks

        # Build coverage count
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in blocks:
            mask = block_masks[bi]
            # Extract set bits
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                cov_count[idx] += 1
                m ^= lb

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Recompute cov_count
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            for bi in blocks:
                mask = block_masks[bi]
                m = mask
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    cov_count[idx] += 1
                    m ^= lb

            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                bi = blocks[i]
                mask = block_masks[bi]
                # Check if all t-subsets in this block are covered by >= 2 blocks
                m = mask
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    blocks.pop(i)
                    improved = True
                    break

        return blocks

    # Greedy with bitsets: pick block with max marginal gain
    def greedy_restart(seed, time_limit):
        random.seed(seed)
        uncovered_mask = (1 << n_tsubs) - 1  # all bits set
        chosen = []

        while uncovered_mask != 0:
            if time.time() > time_limit:
                break

            # Pick a random uncovered t-subset to guide
            # Find an uncovered bit
            m = uncovered_mask
            # Pick a random bit from uncovered
            # Convert to list of set bits is expensive; instead pick random position
            # and check if uncovered, else retry
            # Better: collect uncovered indices
            unc_indices = []
            mm = uncovered_mask
            while mm:
                lb = mm & (-mm)
                unc_indices.append(lb.bit_length() - 1)
                mm ^= lb

            if not unc_indices:
                break

            crit_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]

            # Get all blocks covering this t-subset
            candidates = tsub_to_blocks[crit_idx]
            if not candidates:
                break

            # Find candidate with max gain (number of uncovered t-subsets it covers)
            best_bi = -1
            best_gain = -1
            for bi in candidates:
                gain = (block_masks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi

            if best_bi < 0:
                break

            chosen.append(best_bi)
            uncovered_mask &= ~block_masks[best_bi]

        return chosen

    # Run multiple restarts
    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break

        # Greedy phase
        chosen = greedy_restart(restart, deadline - 15.0)
        if not chosen:
            continue

        # Local search phase
        time_limit_ls = deadline - 5.0
        cleaned = local_search(chosen, time_limit_ls)

        if len(cleaned) < best_count:
            best_count = len(cleaned)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    # If no blocks found, fallback: return empty (shouldn't happen)
    if not best_blocks:
        # Try one more greedy with fixed seed
        chosen = greedy_restart(0, deadline - 5.0)
        if chosen:
            cleaned = local_search(chosen, deadline - 2.0)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    return best_blocks
