import time
import itertools
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    block_masks = [0] * n_blocks
    for bi, block in enumerate(all_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << tsub_index[combo])
        block_masks[bi] = mask

    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            tsub_to_blocks[tsub_index[combo]].append(bi)

    # Precompute for each t-subset, a list of (block_idx, gain_score) sorted by
    # how many other t-subsets that block also covers. This lets greedy pick
    # from a smaller candidate pool.
    # For each t-subset ts, sort its covering blocks by |mask(bi) & ~{ts}|
    # Actually simpler: just sort by total coverage (popcount of mask)
    # But that's the same for all. Instead, use the fact that we want max
    # marginal gain, which depends on uncovered set. So we can't pre-sort fully.
    # Alternative: for each t-subset, keep top 50 blocks by popcount(mask)
    # as a quick-filter pool.
    
    # Precompute popcount for each block mask
    block_popcounts = [m.bit_count() for m in block_masks]
    
    # For each t-subset, sort covering blocks by popcount descending, keep top 80
    tsub_top_blocks = []
    for ts_idx in range(n_tsubs):
        blks = tsub_to_blocks[ts_idx]
        # Sort by popcount descending
        blks_sorted = sorted(blks, key=lambda b: block_popcounts[b], reverse=True)
        tsub_top_blocks.append(blks_sorted[:80])

    best_blocks = []
    best_count = float('inf')

    def local_search(block_indices, time_limit):
        """Remove redundant blocks and try swaps."""
        blocks = list(block_indices)
        if not blocks:
            return blocks

        def compute_cov_count(blk_list):
            cov = np.zeros(n_tsubs, dtype=np.int32)
            for bi in blk_list:
                m = block_masks[bi]
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    cov[idx] += 1
                    m ^= lb
            return cov

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            cov_count = compute_cov_count(blocks)

            # Try removals (in reverse order for stability)
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > time_limit:
                    return blocks
                bi = blocks[i]
                m = block_masks[bi]
                redundant = True
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                    m ^= lb
                if redundant:
                    blocks.pop(i)
                    improved = True
                    break

            # Try swaps: remove block i, add block j that covers some uncovered
            # t-subsets and doesn't leave any t-subset uncovered
            if time.time() > time_limit:
                return blocks
            for i in range(len(blocks)):
                if time.time() > time_limit:
                    return blocks
                bi = blocks[i]
                mask_i = block_masks[bi]
                # Find t-subsets that would become uncovered if we remove bi
                # i.e., those with cov_count == 1
                critical = []
                m = mask_i
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    if cov_count[idx] == 1:
                        critical.append(idx)
                    m ^= lb
                if not critical:
                    continue  # already handled by removal

                # We need a replacement block that covers all critical t-subsets
                # and adds at least one new coverage (or at least doesn't hurt)
                # Find blocks that cover ALL critical t-subsets
                # Use intersection of tsub_to_blocks for critical
                if len(critical) > 3:
                    continue  # too many constraints, skip
                candidate_set = set(tsub_to_blocks[critical[0]])
                for c in critical[1:]:
                    candidate_set &= set(tsub_to_blocks[c])
                    if not candidate_set:
                        break
                if not candidate_set:
                    continue
                # Exclude current blocks
                current_set = set(blocks)
                candidate_set -= current_set
                if not candidate_set:
                    continue
                # Pick best candidate: one that covers most uncovered-after-removal
                # After removing bi, uncovered = critical (those with cov==1)
                # We need the replacement to cover all of critical.
                # Score by how many additional t-subsets it covers (beyond critical)
                best_j = -1
                best_score = -1
                for bj in candidate_set:
                    mask_j = block_masks[bj]
                    # Score: number of t-subsets in mask_j that have cov_count >= 2
                    # (i.e., adding bj creates redundancy, which is good for future removals)
                    # Or simpler: just pick one with highest popcount
                    score = block_popcounts[bj]
                    if score > best_score:
                        best_score = score
                        best_j = bj
                if best_j >= 0:
                    blocks[i] = best_j
                    improved = True
                    break

        return blocks

    def greedy_restart(seed, time_limit):
        random.seed(seed)
        uncovered_mask = (1 << n_tsubs) - 1
        chosen = []

        while uncovered_mask != 0:
            if time.time() > time_limit:
                break

            # Pick a random uncovered t-subset
            # Find uncovered indices efficiently
            unc_indices = []
            mm = uncovered_mask
            while mm:
                lb = mm & (-mm)
                unc_indices.append(lb.bit_length() - 1)
                mm ^= lb

            if not unc_indices:
                break

            crit_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]
            candidates = tsub_top_blocks[crit_idx]
            if not candidates:
                # Fallback to full list
                candidates = tsub_to_blocks[crit_idx]
            if not candidates:
                break

            best_bi = -1
            best_gain = -1
            for bi in candidates:
                gain = (block_masks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi

            if best_bi < 0:
                break

            chosen.append(best_bi)
            uncovered_mask &= ~block_masks[best_bi]

        return chosen

    n_restarts = 200
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break

        chosen = greedy_restart(restart, deadline - 15.0)
        if not chosen:
            continue

        time_limit_ls = deadline - 5.0
        cleaned = local_search(chosen, time_limit_ls)

        if len(cleaned) < best_count:
            best_count = len(cleaned)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    if not best_blocks:
        chosen = greedy_restart(0, deadline - 5.0)
        if chosen:
            cleaned = local_search(chosen, deadline - 2.0)
            best_blocks = [list(all_blocks[bi]) for bi in cleaned]

    return best_blocks
