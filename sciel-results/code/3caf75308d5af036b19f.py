"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using multi-start greedy with
    vectorized gain computation and efficient reduction.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}

    # Precompute for each point, the list of t-subset indices that contain it
    # point_to_subsets[x] = list of indices of t-subsets containing x
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_t_subsets):
        for x in s:
            point_to_subsets[x].append(idx)

    def greedy_cover(uncovered_mask, seed=None, deterministic_first=False):
        """
        Run one greedy pass.
        uncovered_mask: np.array of bools, length N_t. True if uncovered.
        Returns list of blocks (lists of ints).
        """
        if seed is not None:
            random.seed(seed)

        if not np.any(uncovered_mask):
            return []

        blocks = []
        unc = uncovered_mask.copy()
        block_count = 0

        while np.any(unc):
            # Check time
            if time.time() > deadline - 2.0:
                break

            # Pick starting t-subset
            unc_indices = np.where(unc)[0]
            if len(unc_indices) == 0:
                break

            if deterministic_first and block_count == 0:
                # Pick the uncovered subset with the fewest potential extensions
                # (most constrained) - use the one with highest index as heuristic
                start_idx = unc_indices[0]
            else:
                start_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]

            start_subset = all_t_subsets[start_idx]
            block = list(start_subset)
            block_set = set(block)

            # Build a mask of t-subsets that are subsets of the current block
            # as we grow the block, we need to track which uncovered subsets
            # can be completed by adding more points

            while len(block) < k:
                # Check time
                if time.time() > deadline - 2.0:
                    break

                # For each candidate point, compute how many uncovered t-subsets
                # it would cover when added to the block
                # A t-subset is covered by block+{x} if it's a subset of block+{x}
                # and it contains x (since it's not already a subset of block)
                # So we need: t-subset contains x, and t-subset \ {x} is a subset of block

                # Precompute (t-1)-subsets of current block
                block_t_minus_1 = list(itertools.combinations(block, t - 1))

                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block_set]

                # For efficiency, for each candidate x, we check how many of the
                # (t-1)-subsets of block, when combined with x, form an uncovered t-subset
                for x in candidates:
                    gain = 0
                    for rest in block_t_minus_1:
                        cand_tuple = tuple(sorted(rest + (x,)))
                        cid = subset_to_id[cand_tuple]
                        if unc[cid]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None:
                    best_point = candidates[0]

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            block_count += 1

            # Update uncovered mask
            for comb in itertools.combinations(block, t):
                cid = subset_to_id[comb]
                unc[cid] = False

        return blocks

    def reduce_blocks(blocks, seed=None):
        """
        Reduce blocks by checking redundancy using coverage counts.
        Uses multiple passes with random ordering.
        """
        if not blocks:
            return blocks

        if seed is not None:
            random.seed(seed)

        reduced = list(blocks)

        # Precompute t-subset indices for each block
        block_t_subsets = []
        for b in reduced:
            subs = [subset_to_id[comb] for comb in itertools.combinations(b, t)]
            block_t_subsets.append(subs)

        # Build coverage counts
        cover_count = np.zeros(N_t, dtype=np.int32)
        for subs in block_t_subsets:
            for s in subs:
                cover_count[s] += 1

        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices = list(range(len(reduced)))
            random.shuffle(indices)

            for i in indices:
                if i >= len(reduced):
                    continue

                subs = block_t_subsets[i]

                # Decrement counts
                for s in subs:
                    cover_count[s] -= 1

                # Check if any count in subs became 0
                redundant = True
                for s in subs:
                    if cover_count[s] == 0:
                        redundant = False
                        break

                if redundant:
                    # Remove block i
                    reduced.pop(i)
                    block_t_subsets.pop(i)
                    changed = True
                    break
                else:
                    # Restore counts
                    for s in subs:
                        cover_count[s] += 1

        return reduced

    def verify(blocks):
        """Verify that all t-subsets are covered."""
        covered = np.zeros(N_t, dtype=bool)
        for b in blocks:
            for comb in itertools.combinations(b, t):
                covered[subset_to_id[comb]] = True
        return np.all(covered)

    best_blocks = None
    best_count = float('inf')

    initial_mask = np.ones(N_t, dtype=bool)

    restart_count = 0
    # First, do one deterministic run
    blocks = greedy_cover(initial_mask, seed=42, deterministic_first=True)
    if blocks and verify(blocks):
        best_count = len(blocks)
        best_blocks = blocks

    # Now do randomized restarts
    while time.time() < deadline - 10.0:
        restart_count += 1
        seed = restart_count * 12345 + 7
        blocks = greedy_cover(initial_mask, seed=seed)
        if not blocks:
            continue
        if not verify(blocks):
            continue
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        # Try reduction
        if time.time() < deadline - 8.0:
            reduced = reduce_blocks(best_blocks, seed=seed + 1)
            if reduced and verify(reduced):
                if len(reduced) < best_count:
                    best_count = len(reduced)
                    best_blocks = reduced

    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_cover(initial_mask, seed=1)
        best_blocks = blocks

    return best_blocks
# EVOLVE-BLOCK-END
