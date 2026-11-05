"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Enumerate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Enumerate all k-subsets (candidate blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Build incidence matrix: block i covers t-subset j if t_sub j is subset of k_sub i
    # For each t-subset, find which blocks contain it
    # A block (k-subset) contains a t-subset iff all t elements are in the k-subset
    # Precompute: for each t-subset, the set of blocks that contain it
    # This is expensive for large v,k,t but for v=19,k=8,t=5:
    # C(19,5)=11628 t-subsets, C(19,8)=75582 k-subsets
    # For each t-subset, the number of k-subsets containing it is C(19-5, 8-5) = C(14,3) = 364

    # Build a mapping: for each t-subset index, list of block indices that cover it
    # And for each block index, list of t-subset indices it covers
    # Use numpy for vectorized operations where possible

    # For each block, compute which t-subsets it covers
    # block_cov[i] = set of t-subset indices covered by block i
    # t_block_cov[j] = set of block indices covering t-subset j

    # Build t_subset to block coverage mapping
    # For each t-subset, find all k-subsets that contain it
    # A k-subset contains a t-subset iff it contains all t elements plus (k-t) more from the remaining v-t
    t_sub_blocks = [[] for _ in range(n_t)]
    for j, tsub in enumerate(all_t_subs):
        tset = set(tsub)
        remaining = [p for p in range(v) if p not in tset]
        for extra in itertools.combinations(remaining, k - t):
            ksub = tuple(sorted(tsub + extra))
            # Need to find index of ksub in all_k_subs
            # This is slow; let's use a different approach

    # Better approach: build incidence matrix directly using bit masks
    # Represent each subset as a bitmask (v <= 64 fits in uint64)
    # t_sub_masks[j] = bitmask of t-subset j
    # k_sub_masks[i] = bitmask of k-subset i
    # Block i covers t-subset j iff (k_sub_masks[i] & t_sub_masks[j]) == t_sub_masks[j]

    t_sub_masks = np.zeros(n_t, dtype=np.uint64)
    for j, sub in enumerate(all_t_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_sub_masks[j] = mask

    k_sub_masks = np.zeros(n_k, dtype=np.uint64)
    for i, sub in enumerate(all_k_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        k_sub_masks[i] = mask

    # Incidence: inc[i, j] = 1 if block i covers t-subset j
    # For v=19, n_t=11628, n_k=75582: this is 75582*11628 ≈ 878M entries - too large for dense
    # Use sparse representation or compute on-the-fly

    # Alternative: for each t-subset j, precompute list of blocks that cover it
    # C(14,3) = 364 blocks per t-subset on average
    # Total edges: 11628 * 364 ≈ 4.2M - manageable

    # Build t_sub_to_blocks: for each t-subset, list of block indices
    t_sub_to_blocks = [[] for _ in range(n_t)]
    # For each block, determine which t-subsets it covers
    block_to_tsubs = [[] for _ in range(n_k)]

    # For each block i, the t-subsets it covers are all t-subsets of the k-subset
    # Number of t-subsets per block: C(8,5) = 56
    for i, ksub in enumerate(all_k_subs):
        for tsub in itertools.combinations(ksub, t):
            j = t_sub_to_idx[tsub]
            block_to_tsubs[i].append(j)
            t_sub_to_blocks[j].append(i)

    # Precompute flat block-to-tsubs array ONCE (used by all greedy runs)
    max_tsubs_per_block = max(len(lst) for lst in block_to_tsubs)
    block_tsubs_flat = np.full((n_k, max_tsubs_per_block), -1, dtype=np.int32)
    valid_mask = np.zeros((n_k, max_tsubs_per_block), dtype=bool)
    for i in range(n_k):
        lst = block_to_tsubs[i]
        L = len(lst)
        if L > 0:
            block_tsubs_flat[i, :L] = lst
            valid_mask[i, :L] = True
    safe_flat = np.where(block_tsubs_flat >= 0, block_tsubs_flat, 0)

    # Precompute a tuple->index map for fast block ID lookup
    ksub_to_idx = {sub: i for i, sub in enumerate(all_k_subs)}

    def greedy_cover_vectorized(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(n_t, dtype=np.uint8)
        blocks_used = []
        used_blocks_set = set()
        cov_count = np.zeros(n_t, dtype=np.int32)

        while True:
            if time.time() > deadline:
                break

            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break

            first_uncovered = uncovered_idx[0]

            # Compute gains vectorized
            uncovered_flag = (cov_count == 0).astype(np.int32)
            uncovered_vals = uncovered_flag[safe_flat]
            gains = np.where(valid_mask, uncovered_vals, 0).sum(axis=1)

            # Exclude already-used blocks
            if used_blocks_set:
                used_arr = np.fromiter(used_blocks_set, dtype=np.int32, count=len(used_blocks_set))
                gains[used_arr] = 0

            # Small random perturbation for tie-breaking diversity
            if seed is not None and len(used_blocks_set) < n_k:
                rng = np.random.RandomState(seed + len(blocks_used))
                perturbation = rng.randint(0, 2, size=n_k).astype(np.int32)
                gains = gains + perturbation
                if used_blocks_set:
                    perturbation[used_arr] = 0

            best_block = int(np.argmax(gains))
            best_gain = int(gains[best_block])

            if best_gain <= 0:
                break

            blocks_used.append(list(all_k_subs[best_block]))
            used_blocks_set.add(best_block)

            for j in block_to_tsubs[best_block]:
                cov_count[j] += 1
                if cov_count[j] == 1:
                    covered[j] = 1

        return blocks_used

    # Run greedy with multiple seeds, keep best
    blocks = greedy_cover_vectorized(seed=42)

    def is_complete(block_list):
        cov = np.zeros(n_t, dtype=np.uint8)
        for block in block_list:
            bid = ksub_to_idx[tuple(block)]
            for j in block_to_tsubs[bid]:
                cov[j] = 1
        return np.all(cov == 1)

    best_blocks = blocks
    for seed in range(1, 20):
        if time.time() > deadline - 15:
            break
        b = greedy_cover_vectorized(seed=seed)
        if is_complete(b) and len(b) < len(best_blocks):
            best_blocks = b

    blocks = best_blocks

    # Fast reduction pass using precomputed indices
    def try_reduce(block_list):
        if not block_list:
            return block_list

        cov_count = np.zeros(n_t, dtype=np.int32)
        block_ids = []
        for block in block_list:
            bid = ksub_to_idx[tuple(block)]
            block_ids.append(bid)
            for j in block_to_tsubs[bid]:
                cov_count[j] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(block_ids)):
                if time.time() > deadline - 2:
                    break
                bid = block_ids[i]
                redundant = True
                for j in block_to_tsubs[bid]:
                    if cov_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    for j in block_to_tsubs[bid]:
                        cov_count[j] -= 1
                    block_ids.pop(i)
                    block_list.pop(i)
                    changed = True
                    break

        return block_list

    if time.time() < deadline - 10:
        blocks = try_reduce(blocks)
        # Second reduction pass to catch cascading redundancies
        if time.time() < deadline - 5:
            blocks = try_reduce(blocks)

    # Final verification
    if not is_complete(blocks):
        # Emergency fallback: simple greedy
        covered = np.zeros(n_t, dtype=np.uint8)
        blocks = []
        while True:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
            first_uncovered = uncovered_idx[0]
            found = False
            for bid in t_sub_to_blocks[first_uncovered]:
                block = list(all_k_subs[bid])
                if block not in blocks:
                    blocks.append(block)
                    for j in block_to_tsubs[bid]:
                        covered[j] = 1
                    found = True
                    break
            if not found:
                break

    return blocks
# EVOLVE-BLOCK-END
