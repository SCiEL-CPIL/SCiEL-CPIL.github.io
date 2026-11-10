"""Bitset-accelerated greedy covering design C(24,7,3) with fast redundancy removal."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute bitset masks
    # For each t-subset index i, mask of points in that t-subset (int bitmask)
    tsub_point_mask = [0] * n_tsubs
    for i, ts in enumerate(all_tsubs):
        m = 0
        for x in ts:
            m |= (1 << x)
        tsub_point_mask[i] = m

    # For each point x, bitmask of t-subset indices that contain x
    # We need n_tsubs bits per point; use Python int as arbitrary-precision bitset
    point_tsub_mask = [0] * v
    for i, ts in enumerate(all_tsubs):
        bit = 1 << i
        for x in ts:
            point_tsub_mask[x] |= bit

    # For each block (k-subset), precompute the bitmask of t-subsets it covers
    # We'll compute on the fly using point_tsub_mask AND for all points in block
    # But that's expensive. Instead, for a given block, covered_tsub_mask =
    #   AND over x in block of point_tsub_mask[x]
    # This works because a t-subset is covered by block iff all its points are in block,
    # iff for every point x in the t-subset, x is in block. But we need: t-subset s
    # is covered iff s ⊆ block. Using AND of point_tsub_mask[x] for x in block gives
    # the set of t-subsets where EVERY point of the t-subset is in block? No.
    # point_tsub_mask[x] = set of t-subsets containing x.
    # AND over x in block = t-subsets that contain ALL points in block.
    # That's the opposite. We want t-subsets that are subsets of block.
    #
    # Alternative: for each t-subset s, it is covered by block iff s ⊆ block.
    # We can precompute for each t-subset s, the bitmask of points in s (tsub_point_mask[s]).
    # Then s ⊆ block iff (tsub_point_mask[s] & ~block_mask) == 0.
    # But iterating over all t-subsets is 2024 per block, which is fine.
    #
    # Better: for the greedy gain computation, we need for candidate point x:
    #   gain(x) = |{s : s uncovered, x ∈ s, s\{x} ⊆ block}|
    # We can precompute for each (t-1)-subset of block, the t-subsets that extend it.
    # For t=3, (t-1)=2, so for each pair in block, the 3-subsets extending it.
    #
    # Let's use a different approach: represent uncovered as a Python int bitmask
    # of n_tsubs bits. For a block, the covered mask is computed by:
    #   For each t-subset s, check if s ⊆ block.
    # We can precompute for each block (k-subset) its coverage mask lazily.
    #
    # Actually, for v=24, k=7, t=3: n_tsubs = C(24,3) = 2024.
    # Python int with 2024 bits is fine. Bitwise operations are fast.
    #
    # Strategy:
    # - uncovered: Python int, bit i set iff t-subset i is uncovered.
    # - For a completed block b, coverage_mask(b) = bitmask of t-subsets ⊆ b.
    # - To compute coverage_mask(b): for each t-subset s, check s ⊆ b.
    #   Precompute tsub_point_mask[s]. Then s ⊆ b iff (tsub_point_mask[s] & ~b_mask) == 0.
    #   This is 2024 iterations per block, each a simple int AND and compare.
    #
    # - For greedy gain: for candidate x not in block,
    #   gain(x) = popcount(uncovered & coverable_by_x_given_block)
    #   where coverable_by_x_given_block = {s : x ∈ s, s\{x} ⊆ block}
    #   For t=3: s\{x} is a pair. So for each pair (a,b) in block,
    #   the 3-subsets extending it with x are just {a,b,x} if x not in {a,b}.
    #   So coverable_by_x_given_block = union over pairs (a,b) in block of
    #   {t-subset {a,b,x}} if x not in {a,b}.
    #   Since each pair (a,b) extends to exactly one 3-subset with x (if x≠a,b),
    #   and there are C(|block|, 2) pairs, gain(x) ≤ C(|block|,2).
    #   We can precompute for each pair (a,b) and point x, the t-subset index of {a,b,x}.
    #
    # Precompute: pair_point_to_tsub_idx[a][b][x] = index of 3-subset {a,b,x}, or -1 if invalid
    # This is 24*24*24 = 13824 entries, small.

    # Precompute pair->point->tsub index mapping
    # pair_to_tsub = {(a,b): {x: idx}} for a<b, x not in {a,b}
    pair_to_tsub = {}
    for a in range(v):
        for b in range(a + 1, v):
            d = {}
            for x in range(v):
                if x != a and x != b:
                    ts = tuple(sorted((a, b, x)))
                    d[x] = tsub_to_idx[ts]
            pair_to_tsub[(a, b)] = d

    # Precompute coverage mask for a given block (as a frozenset or sorted tuple)
    # We'll cache these.
    block_cover_cache = {}

    def compute_block_cover_mask(block):
        """Return bitmask of t-subsets covered by block."""
        key = tuple(sorted(block))
        if key in block_cover_cache:
            return block_cover_cache[key]
        b_mask = 0
        for x in block:
            b_mask |= (1 << x)
        cover = 0
        for i in range(n_tsubs):
            if (tsub_point_mask[i] & ~b_mask) == 0:
                cover |= (1 << i)
        block_cover_cache[key] = cover
        return cover

    def popcount(x):
        return bin(x).count('1')

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(seed):
        """Run bitset-accelerated greedy covering."""
        rng = random.Random(seed)
        # uncovered: bit i set iff t-subset i is uncovered
        uncovered = (1 << n_tsubs) - 1
        blocks = []

        while uncovered:
            # Pick a random uncovered t-subset to start
            unc_bits = uncovered
            # Get list of uncovered indices
            unc_idx = []
            while unc_bits:
                # Find lowest set bit
                lb = unc_bits & (-unc_bits)
                i = lb.bit_length() - 1
                unc_idx.append(i)
                unc_bits ^= lb
            if not unc_idx:
                break
            start_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            block = list(all_tsubs[start_idx])
            block_set = set(block)

            while len(block) < k:
                # Compute gain for each candidate x not in block
                # For t=3: gain(x) = number of uncovered 3-subsets {x,a,b} where {a,b} ⊆ block
                gains = {}
                for x in all_points:
                    if x in block_set:
                        continue
                    g = 0
                    for i in range(len(block)):
                        for j in range(i + 1, len(block)):
                            a, b = block[i], block[j]
                            # Get t-subset index for {a,b,x}
                            d = pair_to_tsub[(a, b)] if (a, b) in pair_to_tsub else None
                            if d is None:
                                # Try swapped
                                d = pair_to_tsub[(b, a)] if (b, a) in pair_to_tsub else None
                            if d is not None and x in d:
                                idx = d[x]
                                if (uncovered >> idx) & 1:
                                    g += 1
                    gains[x] = g

                if not gains or max(gains.values()) == 0:
                    candidates = [x for x in all_points if x not in block_set]
                    if not candidates:
                        break
                    x = rng.choice(candidates)
                    block.append(x)
                    block_set.add(x)
                else:
                    max_g = max(gains.values())
                    best_candidates = [x for x, g in gains.items() if g == max_g]
                    x = rng.choice(best_candidates)
                    block.append(x)
                    block_set.add(x)

            block.sort()
            blocks.append(block)

            # Update uncovered: remove all t-subsets covered by this block
            cover_mask = compute_block_cover_mask(block)
            uncovered &= ~cover_mask

        return blocks

    def remove_redundant_blocks(blocks, seed=0):
        """Fast redundancy removal using bitset coverage."""
        rng = random.Random(seed)
        blocks = list(blocks)
        # Precompute coverage masks for all blocks
        covers = [compute_block_cover_mask(b) for b in blocks]
        changed = True
        while changed:
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                # Check if block i is redundant: every t-subset in block i is covered by some other block
                # tsub_set_i = set of t-subset indices in block i
                # For each such t-subset, check if it's in covers[j] for some j != i
                # Equivalent: for each t-subset s in block i, exists j != i such that bit s is in covers[j]
                # We can check: for each s in tsub_set_i, (OR of covers[j] for j!=i) has bit s set
                # But OR of all covers except i is expensive. Instead, for each s, scan j.
                # Since |blocks| is small (~76), this is fine.
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[ts]
                    covered = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if (covers[j] >> idx) & 1:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    covers.pop(i)
                    changed = True
                    break
        return blocks

    # Try multiple seeds
    seed = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        if blocks and len(blocks) < best_count:
            blocks2 = remove_redundant_blocks(blocks, seed)
            if len(blocks2) < len(blocks):
                blocks = blocks2
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
        if time.time() > deadline:
            break

    if best_blocks is None:
        blocks = greedy_cover(0)
        best_blocks = blocks

    return best_blocks
