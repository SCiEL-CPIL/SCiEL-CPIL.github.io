import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s kill

    # Map every t-subset to an integer id
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # For each point x, and each (t-1)-tuple S, store the bit of the t-subset S∪{x}
    # masks[x][S] = bit (1 << idx) of the unique t-subset containing x and S
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i in range(t):
            others = s[:i] + s[i+1:]
            x = s[i]
            masks[x][others] = bit

    covered = 0  # Bitset of covered t-subsets
    blocks = []
    block_set = set()

    def popcount(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def build_block_from(seed_points):
        """
        Grow a block from initial set (seed_points) to size k by max-gain greedy.
        Returns the block as a list of ints.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        uncov = ~covered

        # Precompute gain masks for all available candidates
        # gain_masks[x] = OR over (t-1)-subsets of B of masks[x][S]
        gain_masks = {}
        for x in available:
            gm = 0
            for S in itertools.combinations(B, t - 1):
                m = masks[x].get(S)
                if m is not None:
                    gm |= m
            gain_masks[x] = gm

        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in available:
                g = popcount(gain_masks[x] & uncov)
                if g > best_gain:
                    best_gain = g
                    best_x = x

            if best_x == -1:
                break

            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)

            # Update gain masks for remaining candidates
            for x in available:
                gm = gain_masks[x]
                for S in itertools.combinations(B[:-1], t - 2):
                    key = S + (best_x,)
                    m = masks[x].get(key)
                    if m is not None:
                        gm |= m
                gain_masks[x] = gm

        return B

    def total_new_coverage(block):
        """
        Count how many t-subsets in the block are newly covered.
        """
        block_mask = 0
        for s in itertools.combinations(sorted(block), t):
            block_mask |= 1 << subset_to_index[s]
        return popcount(block_mask & ~covered)

    # Main Loop: Strong Greedy with Multiple Randomized Seeds
    all_points = list(range(v))

    while True:
        if time.time() > deadline:
            break

        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices (capped)
        uncovered_indices = []
        m = uncov_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 300:
                break

        if not uncovered_indices:
            break

        best_block = None
        best_gain_total = -1

        # Try multiple seed strategies
        n_seeds = 80

        # 1. Seeds from uncovered triples
        random.shuffle(uncovered_indices)
        selected_uncov = uncovered_indices[:n_seeds // 2]
        for idx in selected_uncov:
            if time.time() > deadline:
                break
            seed = all_subsets[idx]
            B = build_block_from(list(seed))
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B

        # 2. Random seeds
        for _ in range(n_seeds // 2):
            if time.time() > deadline:
                break
            seed = random.sample(range(v), t)
            B = build_block_from(seed)
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B

        # Additional pass with more seeds if time remains
        remaining = deadline - time.time()
        if remaining > 5.0 and best_gain_total >= 0:
            extra_seeds = min(30, int(remaining * 5))
            for _ in range(extra_seeds):
                if time.time() > deadline - 2.0:
                    break
                seed = random.sample(range(v), t)
                B = build_block_from(seed)
                bl = tuple(sorted(B))
                if bl in block_set:
                    continue
                total = total_new_coverage(B)
                if total > best_gain_total:
                    best_gain_total = total
                    best_block = B

        if best_block is None:
            # Try a few more random restarts
            for _ in range(20):
                if time.time() > deadline:
                    break
                seed = random.sample(range(v), t)
                B = build_block_from(seed)
                bl = tuple(sorted(B))
                if bl in block_set:
                    continue
                total = total_new_coverage(B)
                if total > 0:
                    best_block = B
                    best_gain_total = total
                    break
            if best_block is None:
                break

        bl = sorted(best_block)

        # Mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]

        blocks.append(bl)
        block_set.add(tuple(bl))

    return blocks
