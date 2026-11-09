"""Covering design C(19,7,4) – randomized greedy with NumPy acceleration."""
import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute for each point x, which t-subsets contain x
    # point_tsubs[x] = list of indices of t-subsets containing x
    point_tsubs = [[] for _ in range(v)]
    for i, s in enumerate(all_tsubs):
        for x in s:
            point_tsubs[x].append(i)

    def is_complete(covered_mask, n_tsubs):
        """Check if all t-subsets are covered."""
        return np.count_nonzero(covered_mask) >= n_tsubs

    def compute_gain(block_set, uncovered_mask):
        """Compute gain for each candidate point not in block_set.
        Returns array of gains for all v points (0 for points already in block)."""
        gains = np.zeros(v, dtype=np.int32)
        for x in range(v):
            if x in block_set:
                continue
            # Count how many currently-uncovered t-subsets containing x
            # would be covered by adding x to the block
            # A t-subset {x} ∪ rest is covered if 'rest' ⊆ block
            # We need: rest ⊆ block AND the t-subset is currently uncovered
            # Since block has size < k, and t=4, we check t-1=3 subsets of block
            # Actually: for each 3-subset 'rest' of block, check if (rest + x) is uncovered
            if len(block_set) >= t - 1:
                rest_combos = itertools.combinations(block_set, t - 1)
                for rest in rest_combos:
                    tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(tsub)
                    if idx is not None and uncovered_mask[idx]:
                        gains[x] += 1
            else:
                # block too small to form a t-subset with x, gain is 0
                pass
        return gains

    def greedy_cover(rng, alpha, deadline, max_blocks=300):
        """One randomized greedy run. Returns (blocks, covered_mask)."""
        uncovered_mask = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while np.any(uncovered_mask):
            if time.time() >= deadline:
                return blocks, uncovered_mask
            if len(blocks) >= max_blocks:
                return blocks, uncovered_mask

            # Pick a seed t-subset: random uncovered one
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = rng.choice(uncovered_indices.tolist())
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)

            # Mark t-subsets in this block as covered (partial)
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx.get(ts)
                if idx is not None:
                    uncovered_mask[idx] = False

            # Extend block to size k
            while len(block) < k:
                if time.time() >= deadline:
                    return blocks, uncovered_mask

                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break

                gains = compute_gain(block_set, uncovered_mask)
                cand_gains = np.array([gains[x] for x in candidates], dtype=np.float64)

                # Weighted selection
                eps = 0.01
                weights = (cand_gains + eps) ** alpha
                total_w = weights.sum()
                if total_w <= 0:
                    x = rng.choice(candidates)
                else:
                    probs = weights / total_w
                    x = candidates[rng.choice(len(candidates), p=probs)]

                block.append(x)
                block_set.add(x)

                # Mark newly covered t-subsets
                for ts in itertools.combinations(block, t):
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered_mask[idx]:
                        uncovered_mask[idx] = False

            block.sort()
            blocks.append(block)

        return blocks, uncovered_mask

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with different alpha and seeds
    alphas = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    seeds = [42, 123, 456, 789, 1011, 2020, 3030, 4040, 5050, 6060,
             7070, 8080, 9090, 11111, 22222, 33333, 44444, 55555]

    for alpha in alphas:
        for seed in seeds:
            if time.time() >= deadline - 3.0:
                break
            rng = random.Random(seed + int(alpha * 1000))
            blocks, mask = greedy_cover(rng, alpha, deadline)
            if blocks and not np.any(mask):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = blocks
                    if best_count <= 140:
                        return best_blocks

    if best_blocks is not None:
        return best_blocks

    # Fallback: deterministic greedy
    rng = random.Random(0)
    blocks, mask = greedy_cover(rng, 0.0, deadline)
    if blocks:
        return blocks
    return []
