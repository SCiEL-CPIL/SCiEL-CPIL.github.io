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

    # Convert to numpy arrays for fast access
    block_to_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsubs]
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    def greedy_cover_vectorized(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(n_t, dtype=np.uint8)
        blocks_used = []
        used_blocks_set = set()

        # Coverage count: how many selected blocks cover each t-subset
        cov_count = np.zeros(n_t, dtype=np.int32)

        while True:
            # Find first uncovered t-subset
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break

            if time.time() > deadline:
                break

            first_uncovered = uncovered_idx[0]

            # Find best block: maximizes number of newly covered t-subsets
            # For each candidate block not yet used, compute gain
            # gain(i) = number of t-subsets in block_to_tsubs[i] that are not yet covered
            # Vectorized: for all blocks, compute sum of (1 - covered[block_to_tsubs[i]])

            # Process in chunks to avoid memory issues
            best_gain = -1
            best_block = -1

            # We can compute gains for all blocks at once using the incidence structure
            # For each block i, gain = sum over j in block_to_tsubs[i] of (1 - covered[j])
            # This is equivalent to: for each block i, gain = len(block_to_tsubs[i]) - sum(covered[block_to_tsubs[i]])

            # Precompute: for each t-subset j, cov_count[j] tells us how many blocks cover it
            # A t-subset is "newly covered" by block i if cov_count[j] == 0 and j is in block_to_tsubs[i]

            # Vectorized gain computation:
            # gains[i] = sum_{j in block_to_tsubs[i]} (cov_count[j] == 0)
            # We can compute this by: for each t-subset j with cov_count[j]==0, add 1 to all blocks in t_sub_to_blocks[j]

            # Efficient approach: iterate over uncovered t-subsets and accumulate
            # But that's O(n_uncovered * 364) which could be slow

            # Better: precompute gains using the block_to_tsubs structure
            # For all blocks at once:
            # Create a matrix where row i is block i, and we need to sum covered==0 over its t-subsets

            # Since each block has exactly C(8,5)=56 t-subsets, we can do:
            # gains = np.array([np.sum(covered[block_to_tsubs_np[i]] == 0) for i in range(n_k)])
            # But this is a Python loop over 75582 blocks - might be slow

            # Alternative: use the fact that gains can be computed incrementally
            # Actually, let's just do it with numpy by building a sparse incidence

            # For now, use a semi-vectorized approach:
            # Group blocks by their t-subset coverage pattern
            # Actually, simplest: compute gains for all blocks using numpy broadcasting in chunks

            # Let's precompute the block_to_tsubs as a fixed-size array with padding
            max_tsubs_per_block = max(len(lst) for lst in block_to_tsubs)
            # For v=19,k=8,t=5: C(8,5) = 56
            block_tsubs_flat = np.full((n_k, max_tsubs_per_block), -1, dtype=np.int32)
            block_tsubs_lens = np.zeros(n_k, dtype=np.int32)
            for i in range(n_k):
                lst = block_to_tsubs[i]
                L = len(lst)
                block_tsubs_flat[i, :L] = lst
                block_tsubs_lens[i] = L

            # Compute gains: for each block, count how many of its t-subsets are uncovered
            # covered_flat = covered[block_tsubs_flat] -> shape (n_k, max_tsubs_per_block)
            # But we need to ignore padding (-1 entries)
            # mask: where block_tsubs_flat >= 0
            valid_mask = block_tsubs_flat >= 0

            # Get covered status for each (block, t-subset) pair
            # covered[block_tsubs_flat] would index with -1 which wraps around - bad
            # Use np.where or handle padding

            # Alternative: compute gains differently
            # gains[i] = number of j in block_to_tsubs[i] with covered[j] == 0
            # = number of j in block_to_tsubs[i] with cov_count[j] == 0 (since we only add blocks, cov_count==0 means uncovered)

            # Use cov_count == 0 as the "uncovered" indicator
            uncovered_flag = (cov_count == 0).astype(np.int32)

            # For each block i, gain = sum of uncovered_flag[j] for j in block_to_tsubs[i]
            # Vectorized:
            # Create an array of shape (n_k, max_tsubs_per_block) where entry (i,j) = uncovered_flag[block_tsubs_flat[i,j]] if valid
            # Then sum along axis 1, masking invalid entries

            # To avoid the -1 indexing issue, replace -1 with a dummy index
            safe_flat = np.where(block_tsubs_flat >= 0, block_tsubs_flat, 0)
            uncovered_vals = uncovered_flag[safe_flat]  # shape (n_k, max_tsubs_per_block)
            # Zero out invalid positions
            uncovered_vals = np.where(valid_mask, uncovered_vals, 0)
            gains = uncovered_vals.sum(axis=1)

            # Exclude already-used blocks
            for bi in used_blocks_set:
                gains[bi] = 0

            # Find best block
            best_block = int(np.argmax(gains))
            best_gain = int(gains[best_block])

            if best_gain <= 0:
                # No improvement possible with remaining blocks - shouldn't happen if we have a valid covering
                break

            # Select this block
            blocks_used.append(list(all_k_subs[best_block]))
            used_blocks_set.add(best_block)

            # Update coverage
            for j in block_to_tsubs[best_block]:
                cov_count[j] += 1
                if cov_count[j] == 1:
                    covered[j] = 1

        return blocks_used

    # Run greedy
    blocks = greedy_cover_vectorized(seed=42)

    # Check if complete
    def is_complete(blocks):
        cov = np.zeros(n_t, dtype=np.uint8)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[t_sub_to_idx[sub]] = 1
        return np.all(cov == 1)

    if not is_complete(blocks):
        # Try different seeds
        for seed in range(1, 10):
            if time.time() > deadline - 5:
                break
            b = greedy_cover_vectorized(seed=seed)
            if is_complete(b) and len(b) < len(blocks):
                blocks = b

    # Reduction pass: remove redundant blocks using coverage counts
    def try_reduce(blocks):
        if not blocks:
            return blocks

        # Build coverage count
        cov_count = np.zeros(n_t, dtype=np.int32)
        block_indices = []  # index of each block in all_k_subs
        for block in blocks:
            block_indices.append(all_k_subs.index(tuple(block)))
            for j in block_to_tsubs[all_k_subs.index(tuple(block))]:
                cov_count[j] += 1

        # Actually, let's rebuild more carefully
        cov_count = np.zeros(n_t, dtype=np.int32)
        block_ids = []
        for block in blocks:
            bid = all_k_subs.index(tuple(block))
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
                # Check if any t-subset is only covered by this block
                redundant = True
                for j in block_to_tsubs[bid]:
                    if cov_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for j in block_to_tsubs[bid]:
                        cov_count[j] -= 1
                    block_ids.pop(i)
                    blocks.pop(i)
                    changed = True
                    break  # restart

        return blocks

    if time.time() < deadline - 10:
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
            # Find a block covering first_uncovered
            found = False
            for bid in t_sub_to_blocks[first_uncovered]:
                # Check if this block is new
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
