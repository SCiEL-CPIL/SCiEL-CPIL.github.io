"""Rewritten program for covering design C(v,k,t) with a stronger greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t): a set of k-subsets (blocks) of {0..v-1}
    such that every t-subset is contained in at least one block.
    Minimize the number of blocks.
    """
    deadline = time.time() + 112.0  # Safety margin before 120s limit

    # 1. Precompute all t-subsets and assign an index to each
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # 2. Precompute the bitmask for every possible k-subset (block)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)

    # Build block_masks as a list of Python ints (bitmasks)
    block_masks = []
    for block in all_k_subsets:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks.append(mask)

    # 3. Bit-slice representation: each of n_t t-subsets is a column of bits.
    #    bits[b, j] = 1 iff t-subset j is covered by block b.
    n_words = (n_t + 63) // 64
    # bits shape: (n_blocks, n_words, 64) uint8
    bits = np.zeros((n_blocks, n_words, 64), dtype=np.uint8)
    for b, mask in enumerate(block_masks):
        for w in range(n_words):
            word = (mask >> (64 * w)) & 0xFFFFFFFFFFFFFFFF
            # unpack 64 bits into 64 uint8
            for bit in range(64):
                bits[b, w, bit] = (word >> bit) & 1

    # Precompute per-block total coverage (for tie-breaking / sanity)
    block_total = np.zeros(n_blocks, dtype=np.int32)
    for b, mask in enumerate(block_masks):
        block_total[b] = mask.bit_count()

    def run_greedy(seed):
        rng = random.Random(seed)
        # uncovered: (n_words, 64) uint8, 1 = not yet covered
        unc = np.ones((n_words, 64), dtype=np.uint8)
        # Mask out bits beyond n_t in the last word
        extra = 64 * n_words - n_t
        if extra > 0:
            unc[n_words - 1, 64 - extra:] = 0

        sel = []
        used = np.zeros(n_blocks, dtype=bool)

        while True:
            if time.time() > deadline - 2.0:
                return None

            # Vectorized gain: for each unused block, count bits set in (bits[b] & unc)
            # gain[b] = sum over w,bit of bits[b,w,bit] * unc[w,bit]
            if not used.any():
                gains = np.sum(bits * unc[None, :, :], axis=(1, 2)).astype(np.int32)
            else:
                gains = np.zeros(n_blocks, dtype=np.int32)
                # Only compute for unused blocks to save time
                unused_mask = ~used
                if unused_mask.any():
                    # (n_blocks, n_words, 64) * (1, n_words, 64) -> sum
                    prod = bits * unc[None, :, :]
                    gains[unused_mask] = np.sum(prod[unused_mask], axis=(1, 2)).astype(np.int32)

            max_gain = gains.max() if len(gains) else 0
            if max_gain <= 0:
                break

            # Collect indices with max gain, pick one at random
            best_idx = np.where(gains == max_gain)[0]
            chosen = int(rng.choice(best_idx))

            sel.append(all_k_subsets[chosen])
            used[chosen] = True
            unc &= ~bits[chosen]

            # Early exit if nothing left
            if not unc.any():
                break

        return sel

    def verify(blocks):
        """Return True if every t-subset is covered by at least one block."""
        covered_mask = 0
        for block in blocks:
            covered_mask |= block_masks[block_index_map[tuple(block)]]
        return covered_mask == (1 << n_t) - 1

    # Map from block tuple to index for verification
    block_index_map = {tuple(b): i for i, b in enumerate(all_k_subsets)}

    best_blocks = None
    best_count = float('inf')

    # First run: deterministic (seed=0)
    res = run_greedy(0)
    if res:
        best_blocks = res
        best_count = len(res)

    # Try many more seeds now that each greedy step is vectorized
    seed = 1
    while time.time() < deadline - 20.0:
        res = run_greedy(seed)
        if res and len(res) < best_count:
            best_blocks = res
            best_count = len(res)
        seed += 1

    # Local search: try to remove a block and see if coverage is still complete.
    # If not, try to find a replacement block that covers the lost t-subsets
    # with fewer additional blocks than the removed one (net reduction).
    if best_blocks is not None and time.time() < deadline - 10.0:
        current = list(best_blocks)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            n = len(current)
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                # Try removing block i
                trial = current[:i] + current[i+1:]
                # Check if still valid
                covered_mask = 0
                for block in trial:
                    covered_mask |= block_masks[block_index_map[tuple(block)]]
                if covered_mask == (1 << n_t) - 1:
                    # Valid without block i!
                    current = trial
                    improved = True
                    break
                # Otherwise, try to find a single replacement block that covers
                # the missing t-subsets (i.e., covers all t-subsets only covered by block i)
                missing = (~covered_mask) & ((1 << n_t) - 1)
                if missing:
                    # Find blocks that cover all missing t-subsets
                    # i.e., block_masks[j] & missing == missing
                    found = False
                    for j in range(n_blocks):
                        if block_masks[j] & missing == missing:
                            # Check j is not already in trial
                            if j not in [block_index_map[tuple(b)] for b in trial]:
                                new_trial = trial + [all_k_subsets[j]]
                                current = new_trial
                                improved = True
                                found = True
                                break
                    if found:
                        break

        if len(current) < best_count:
            best_blocks = current
            best_count = len(current)

    if best_blocks is None:
        return []

    return best_blocks
# EVOLVE-BLOCK-END
