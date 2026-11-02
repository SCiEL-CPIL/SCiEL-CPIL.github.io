# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _tsub_index(t, v):
    """Return an array mapping each t-subset (as a tuple) to a unique integer index."""
    from math import comb
    n = comb(v, t)
    idx = np.empty(n, dtype=np.int64)
    i = 0
    for c in itertools.combinations(range(v), t):
        idx[i] = i
        i += 1
    tsub_to_idx = {}
    i = 0
    for c in itertools.combinations(range(v), t):
        tsub_to_idx[frozenset(c)] = i
        i += 1
    return idx, tsub_to_idx, n

def _block_mask(block, t, tsub_to_idx):
    """Return a boolean mask of t-subsets covered by a block."""
    n = len(tsub_to_idx)
    mask = np.zeros(n, dtype=bool)
    for c in itertools.combinations(block, t):
        mask[tsub_to_idx[frozenset(c)]] = True
    return mask

def _greedy_construct(v, k, t, rng, deadline, tsub_to_idx, n_tsubs):
    """Greedy covering with vectorized gain computation."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    blocks = []
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}
    idx_to_tsub = all_tsubs

    # Precompute for each point x, a dict mapping frozenset(rest) -> index
    # where rest is a (t-1)-subset, and index is the t-subset {x} + rest
    point_tsub_lookup = []
    for x in range(v):
        lookup = {}
        for i, c in enumerate(all_tsubs):
            if x in c:
                rest = tuple(sorted(set(c) - {x}))
                lookup[frozenset(rest)] = i
        point_tsub_lookup.append(lookup)

    def _greedy(v, k, t, rng, deadline, uncovered_init=None):
        """Greedy covering with vectorized gain."""
        if uncovered_init is not None:
            uncovered = uncovered_init.copy()
        else:
            uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []

        while uncovered.any():
            if deadline and time.time() > deadline:
                break

            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = int(rng.choice(uncovered_indices))
            seed_tsub = idx_to_tsub[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)

            t_minus_1 = t - 1

            while len(block) < k:
                if deadline and time.time() > deadline:
                    break

                t1_subsets = list(itertools.combinations(block, t_minus_1))
                t1_keys = [frozenset(s) for s in t1_subsets]

                gains = np.zeros(v, dtype=np.int64)

                for x in range(v):
                    if x in block_set:
                        continue
                    lookup = point_tsub_lookup[x]
                    gain = 0
                    for key in t1_keys:
                        idx = lookup.get(key, -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    gains[x] = gain

                mask = np.ones(v, dtype=bool)
                for x in block:
                    mask[x] = False
                valid_gains = gains[mask]
                if len(valid_gains) > 0:
                    max_gain = np.max(valid_gains)
                    if max_gain > 0:
                        candidates = np.where(mask & (gains == max_gain))[0]
                        best_point = int(rng.choice(candidates))
                    else:
                        candidates = np.where(mask)[0]
                        best_point = int(rng.choice(candidates))
                else:
                    break

                block.append(best_point)
                block_set.add(best_point)

            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered[idx] = False

        return [list(b) for b in blocks]

    def _local_search(blocks, deadline):
        """Remove redundant blocks and try replacements."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)

        if n == 0:
            return blocks

        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[frozenset(c)]] = True
            block_masks.append(mask)
            cov_count += mask.astype(np.int64)

        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if deadline and time.time() > deadline:
                    return blocks
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    cov_count -= mask.astype(np.int64)
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        # Phase 2: Try to replace blocks
        for i in range(len(blocks)):
            if deadline and time.time() > deadline:
                break

            mask_i = block_masks[i]
            unique = mask_i & (cov_count == 1)
            if not unique.any():
                continue

            unique_indices = np.where(unique)[0]

            union_pts = set()
            for idx in unique_indices:
                union_pts.update(idx_to_tsub[idx])

            if len(union_pts) <= k:
                candidates = list(union_pts)
                extra_pool = [x for x in range(v) if x not in union_pts]
                while len(candidates) < k and extra_pool:
                    candidates.append(extra_pool.pop(rng.randint(0, len(extra_pool)-1)))

                if len(candidates) >= k:
                    for attempt in range(100):
                        if deadline and time.time() > deadline:
                            break
                        if len(candidates) == k:
                            new_block = sorted(candidates)
                        else:
                            new_block = sorted(rng.sample(candidates, k))

                        new_mask = np.zeros(n_tsubs, dtype=bool)
                        for c in itertools.combinations(new_block, t):
                            new_mask[tsub_to_idx[frozenset(c)]] = True

                        if np.all(new_mask[unique_indices]):
                            cov_count -= mask_i.astype(np.int64)
                            cov_count += new_mask.astype(np.int64)
                            blocks[i] = new_block
                            block_masks[i] = new_mask
                            break

        # Phase 3: Another pass of redundant block removal
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if deadline and time.time() > deadline:
                    return blocks
                mask = block_masks[i]
                if not np.any(cov_count[mask] <= 1):
                    cov_count -= mask.astype(np.int64)
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        return blocks

    # Run greedy with multiple restarts
    best_blocks = None
    best_len = float('inf')

    rng0 = random.Random(42)
    baseline = _greedy(v, k, t, rng0, deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)

    restart = 0
    while time.time() < deadline - 25:
        restart += 1
        rng = random.Random(restart * 12345 + 7)
        blocks = _greedy(v, k, t, rng, deadline)

        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    if best_blocks and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, deadline)

    return best_blocks if best_blocks else []

# EVOLVE-BLOCK-END
