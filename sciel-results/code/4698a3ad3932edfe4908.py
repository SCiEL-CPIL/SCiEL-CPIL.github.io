"""Improved program for covering design C(19,8,5) with faster greedy and reduction."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Enumerate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Enumerate all k-subsets (candidate blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Build incidence: for each block, which t-subsets it covers
    # For v=19,k=8,t=5: each block covers C(8,5)=56 t-subsets
    block_to_tsubs = [[] for _ in range(n_k)]
    t_sub_to_blocks = [[] for _ in range(n_t)]

    for i, ksub in enumerate(all_k_subs):
        for tsub in itertools.combinations(ksub, t):
            j = t_sub_to_idx[tsub]
            block_to_tsubs[i].append(j)
            t_sub_to_blocks[j].append(i)

    # Convert to numpy arrays for fast access
    block_to_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsubs]

    def greedy_cover(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(n_t, dtype=np.uint8)
        blocks_used = []
        used_blocks_set = set()

        # Coverage count: how many selected blocks cover each t-subset
        cov_count = np.zeros(n_t, dtype=np.int32)

        # Precompute the block-tsub incidence as a flat array for vectorized gain computation
        max_tsubs_per_block = max(len(lst) for lst in block_to_tsubs)
        block_tsubs_flat = np.full((n_k, max_tsubs_per_block), -1, dtype=np.int32)
        block_tsubs_lens = np.zeros(n_k, dtype=np.int32)
        for i in range(n_k):
            lst = block_to_tsubs[i]
            L = len(lst)
            block_tsubs_flat[i, :L] = lst
            block_tsubs_lens[i] = L

        valid_mask = block_tsubs_flat >= 0
        safe_flat = np.where(valid_mask, block_tsubs_flat, 0)

        while True:
            # Find first uncovered t-subset
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break

            if time.time() > deadline - 5:
                break

            # Compute gains for all blocks: number of newly covered t-subsets
            uncovered_flag = (cov_count == 0).astype(np.int32)
            uncovered_vals = uncovered_flag[safe_flat]
            uncovered_vals = np.where(valid_mask, uncovered_vals, 0)
            gains = uncovered_vals.sum(axis=1)

            # Exclude already-used blocks
            for bi in used_blocks_set:
                gains[bi] = 0

            # Find best block
            best_block = int(np.argmax(gains))
            best_gain = int(gains[best_block])

            if best_gain <= 0:
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

    # Run greedy with initial seed
    blocks = greedy_cover(seed=42)

    # Check if complete
    def is_complete(blocks):
        cov = np.zeros(n_t, dtype=np.uint8)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[t_sub_to_idx[sub]] = 1
        return np.all(cov == 1)

    # Try a few more seeds if time permits
    if not is_complete(blocks) or time.time() < deadline - 30:
        best_blocks = blocks
        best_len = len(blocks)
        for seed in range(1, 20):
            if time.time() > deadline - 15:
                break
            b = greedy_cover(seed=seed)
            if is_complete(b) and len(b) < best_len:
                best_blocks = b
                best_len = len(b)
        blocks = best_blocks

    # Reduction pass: remove redundant blocks
    def try_reduce(blocks):
        if not blocks:
            return blocks

        # Build coverage count
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
                if time.time() > deadline - 5:
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
