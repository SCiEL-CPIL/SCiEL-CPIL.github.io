"""Covering design C(19,8,5): fast greedy with incremental gain + local refinement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 118.0


def construct(v, k, t):
    start_time = time.time()

    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # cov[bi, ti] = 1 if t-subset ti in block bi
    cov = np.zeros((n_blocks, n_t), dtype=np.uint8)
    for bi, blk in enumerate(all_blocks):
        for ts in itertools.combinations(blk, t):
            cov[bi, t_idx[ts]] = 1

    # Block masks as Python ints
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        m = 0
        for ts in itertools.combinations(blk, t):
            m |= (1 << t_idx[ts])
        block_masks[bi] = m

    # For each t-subset, list of blocks covering it (for incremental updates)
    t_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_blocks):
        for ti in np.where(cov[bi])[0]:
            t_to_blocks[ti].append(bi)

    full_mask = (1 << n_t) - 1

    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        chosen = []

        # Track which t-subsets are uncovered as a set of indices for fast slicing
        uncovered_set = set(range(n_t))
        # gain[bi] = number of uncovered t-subsets block bi covers
        gain = np.array([c.sum() for c in cov], dtype=np.int32)

        while uncovered_set:
            if time.time() - start_time > DEADLINE_SECONDS - 8:
                return None

            best_gain = gain.max()
            if best_gain == 0:
                break

            candidates = np.where(gain == best_gain)[0]
            bi = int(rng.choice(candidates))
            chosen.append(bi)

            # Update: for each t-subset in block bi, remove from uncovered and decrement gains
            blk = all_blocks[bi]
            covered_tis = [t_idx[ts] for ts in itertools.combinations(blk, t)]
            for ti in covered_tis:
                if ti not in uncovered_set:
                    continue
                uncovered_set.discard(ti)
                # Decrement gain for all blocks that cover ti
                for other_bi in t_to_blocks[ti]:
                    gain[other_bi] -= 1

            # Update uncovered integer (for final check)
            for ti in covered_tis:
                uncovered &= ~(1 << ti)

        return chosen

    def refine(block_indices):
        if not block_indices:
            return block_indices
        n = len(block_indices)
        total = 0
        for bi in block_indices:
            total |= block_masks[bi]
        if total != full_mask:
            return block_indices

        improved = True
        while improved:
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                return block_indices
            improved = False
            order = list(range(len(block_indices)))
            random.shuffle(order)
            for i in order:
                if time.time() - start_time > DEADLINE_SECONDS - 3:
                    return block_indices
                u = 0
                for j in range(len(block_indices)):
                    if j != i:
                        u |= block_masks[block_indices[j]]
                if u == full_mask:
                    del block_indices[i]
                    improved = True
                    break
        return block_indices

    best_solution = None
    best_size = float('inf')
    seed = 0

    while time.time() - start_time < DEADLINE_SECONDS - 10:
        sol = run_greedy(seed)
        if sol is not None and len(sol) > 0:
            sol = refine(sol)
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = [all_blocks[bi] for bi in sol]
        seed += 1

    if best_solution is None:
        sol = run_greedy(0)
        if sol is not None:
            return [all_blocks[bi] for bi in sol]
        return []

    return best_solution
# EVOLVE-BLOCK-END
