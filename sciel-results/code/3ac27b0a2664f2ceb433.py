"""Combined covering design C(19,8,5) with bitset acceleration and local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0


def construct(v, k, t):
    start_time = time.time()

    # Precompute all t-subsets and index them
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}

    # For each point x, precompute the list of t-subset indices that contain x
    subsets_with = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for x in s:
            subsets_with[x].append(i)

    # For fast gain computation: for a block B and candidate x,
    # gain = number of uncovered t-subsets that are subsets of B ∪ {x} and contain x
    # This equals: number of (t-1)-subsets R of B such that R ∪ {x} is an uncovered t-subset

    def compute_gain(block, x, uncovered_mask):
        """Count uncovered t-subsets covered by adding x to block."""
        gain = 0
        # A t-subset of block ∪ {x} that contains x corresponds to a (t-1)-subset of block
        for rest in itertools.combinations(block, t - 1):
            s = tuple(sorted(rest + (x,)))
            si = idx_map.get(s, -1)
            if si >= 0 and uncovered_mask[si]:
                gain += 1
        return gain

    def block_mask(block):
        """Return a numpy bool array indicating which t-subsets are covered by block."""
        mask = np.zeros(n_t, dtype=bool)
        for comb in itertools.combinations(block, t):
            mask[idx_map[tuple(comb)]] = True
        return mask

    def greedy_build_block(uncovered_mask, block_seed=None):
        """Greedily build a block of size k maximizing coverage of uncovered t-subsets."""
        if block_seed is not None:
            block = list(block_seed)
        else:
            # Pick a random uncovered t-subset as seed
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                return None
            start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            block = list(all_t_subsets[start_idx])

        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = compute_gain(block, x, uncovered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        return block

    def local_improvement(blocks, uncovered_mask, max_iter=50):
        """Try to improve the covering by swapping points in blocks."""
        if not blocks:
            return blocks, uncovered_mask

        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break

            # Recompute uncovered mask
            uncovered_mask = np.ones(n_t, dtype=bool)
            for b in blocks:
                uncovered_mask &= ~block_mask(b)

            # Try to improve each block by swapping one point
            for bi in range(len(blocks)):
                if time.time() - start_time > DEADLINE_SECONDS - 10:
                    return blocks, uncovered_mask
                block = list(blocks[bi])
                for si in range(len(block)):
                    for x in range(v):
                        if x in block:
                            continue
                        # Try swapping block[si] with x
                        new_block = block[:]
                        new_block[si] = x
                        new_block.sort()
                        new_mask = block_mask(new_block)
                        old_mask = block_mask(block)

                        # Check if new block covers more uncovered subsets
                        new_covered = (new_mask & uncovered_mask).sum()
                        # Also check if removing old block's unique coverage hurts
                        # (other blocks may cover those)
                        # Simple heuristic: just check if new block covers at least as much
                        if new_covered > (old_mask & uncovered_mask).sum():
                            blocks[bi] = new_block
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Final recomputation
        uncovered_mask = np.ones(n_t, dtype=bool)
        for b in blocks:
            uncovered_mask &= ~block_mask(b)
        return blocks, uncovered_mask

    # Main construction loop
    uncovered_mask = np.ones(n_t, dtype=bool)
    blocks = []

    while uncovered_mask.any():
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break

        # Pick a random uncovered t-subset to seed the block
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        start_subset = all_t_subsets[start_idx]

        block = greedy_build_block(uncovered_mask, block_seed=list(start_subset))
        if block is None:
            break

        mask = block_mask(block)
        if (mask & uncovered_mask).any():
            blocks.append(block)
            uncovered_mask &= ~mask

    # Local improvement phase
    if time.time() - start_time < DEADLINE_SECONDS - 15:
        blocks, uncovered_mask = local_improvement(blocks, uncovered_mask, max_iter=30)

    # If still uncovered, complete with deterministic greedy
    if uncovered_mask.any():
        while uncovered_mask.any():
            if time.time() - start_time > DEADLINE_SECONDS - 2:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = uncovered_indices[0]
            start_subset = all_t_subsets[start_idx]
            block = greedy_build_block(uncovered_mask, block_seed=list(start_subset))
            if block is None:
                break
            mask = block_mask(block)
            blocks.append(block)
            uncovered_mask &= ~mask

    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
# EVOLVE-BLOCK-END
