"""Bitmask-accelerated multi-start greedy + removal for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_tsubsets)}

    # Precompute all k-blocks and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_mask[i] = integer bitmask of t-subsets covered by block i
    block_mask = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for combo in itertools.combinations(blk, t):
            idx = subset_to_idx[frozenset(combo)]
            mask |= (1 << idx)
        block_mask[bi] = mask

    # Precompute: for each t-subset, list of block indices that cover it
    t_sub_to_blocks = [[] for _ in range(n_tsubsets)]
    for bi, mask in enumerate(block_mask):
        mm = mask
        while mm:
            lb = mm & (-mm)
            si = lb.bit_length() - 1
            t_sub_to_blocks[si].append(bi)
            mm ^= lb

    def popcount(x):
        """Count set bits in integer x."""
        return bin(x).count('1')

    def greedy_cover_fast():
        """Fast greedy using precomputed structures and bitmask ops."""
        uncovered_mask = (1 << n_tsubsets) - 1
        blocks = []
        used = set()

        while uncovered_mask:
            # Pick random uncovered t-subset
            # Get some uncovered indices
            uncov_list = []
            mm2 = uncovered_mask
            while mm2 and len(uncov_list) < 50:
                lb = mm2 & (-mm2)
                uncov_list.append(lb.bit_length() - 1)
                mm2 ^= lb
            seed_idx = rng.choice(uncov_list)

            # Find best unused block from candidates
            candidates = t_sub_to_blocks[seed_idx]
            # Shuffle a copy for randomization
            cands = candidates[:]
            rng.shuffle(cands)

            best_bi = None
            best_gain = -1
            checked = 0
            for bi in cands:
                if bi in used:
                    continue
                checked += 1
                gain = popcount(uncovered_mask & block_mask[bi])
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if gain == popcount(uncovered_mask):
                        break
                if checked >= 80:
                    break

            if best_bi is None:
                # All top candidates used; pick any unused
                for bi in candidates:
                    if bi not in used:
                        best_bi = bi
                        break
                if best_bi is None:
                    break

            used.add(best_bi)
            blocks.append(list(all_blocks[best_bi]))
            uncovered_mask &= ~block_mask[best_bi]

        return blocks

    def remove_redundant(blocks):
        """Remove redundant blocks using bitmask representation."""
        if not blocks:
            return blocks

        # Build block index map for O(1) lookup
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}

        # Compute coverage count per t-subset
        counts = [0] * n_tsubsets
        for b in blocks:
            bi = block_to_idx.get(tuple(b), -1)
            if bi < 0:
                continue
            mask = block_mask[bi]
            mm = mask
            while mm:
                lb = mm & (-mm)
                si = lb.bit_length() - 1
                counts[si] += 1
                mm ^= lb

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                bi = block_to_idx.get(tuple(blocks[i]), -1)
                if bi < 0:
                    continue
                mask = block_mask[bi]
                # Check if all covered subsets have count > 1
                mm = mask
                redundant = True
                while mm:
                    lb = mm & (-mm)
                    si = lb.bit_length() - 1
                    if counts[si] <= 1:
                        redundant = False
                        break
                    mm ^= lb
                if redundant:
                    mm = mask
                    while mm:
                        lb = mm & (-mm)
                        si = lb.bit_length() - 1
                        counts[si] -= 1
                        mm ^= lb
                    blocks.pop(i)
                    changed = True
                    break

        return blocks

    # Multi-start greedy
    best_blocks = None
    best_count = float('inf')
    n_starts = 0

    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 5:
            break

        blocks = greedy_cover_fast()
        blocks = remove_redundant(blocks)
        n_starts += 1

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

        # Periodically reseed for diversity
        if n_starts % 15 == 0:
            rng.seed(rng.randint(0, 10**9))

    if best_blocks is None:
        blocks = greedy_cover_fast()
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
