"""Covering design C(v,k,t): Lagrangian-guided greedy set cover with bitmask
coverage tracking, redundancy minimalization, and 1-for-1 swap local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 130.0  # safety margin vs 150s kill

    # --- precompute t-subset index table ---
    all_tsubs = list(itertools.combinations(range(v), t))
    n_t = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # --- precompute all k-subsets and their t-subset index lists ---
    all_ksubs = list(itertools.combinations(range(v), k))
    n_k = len(all_ksubs)
    ksub_tsets = []
    for comb in all_ksubs:
        tset = []
        for c in itertools.combinations(comb, t):
            tset.append(tsub_to_idx[c])
        ksub_tsets.append(tset)

    # --- bitmask for each k-subset: bit i set if t-subset i is covered ---
    ksub_masks = [0] * n_k
    for i, tset in enumerate(ksub_tsets):
        mask = 0
        for idx in tset:
            mask |= (1 << idx)
        ksub_masks[i] = mask

    # --- Lagrangian-guided greedy set cover with bitmask ---
    uncovered_mask = (1 << n_t) - 1
    remaining = n_t

    # Dual weights: initially uniform
    w = np.ones(n_t, dtype=np.float64)

    blocks = []
    block_indices = []  # indices into all_ksubs
    used = np.zeros(n_k, dtype=np.int8)

    max_rounds = 300
    for rnd in range(max_rounds):
        if time.time() > deadline - 30.0:  # leave 30s for post-processing
            break
        if remaining == 0:
            break

        # Compute weighted gain for each unused k-subset
        # gain[i] = sum of w[j] for uncovered t-subsets j in k-subset i
        best_idx = -1
        best_gain = 0.0

        for i in range(n_k):
            if used[i]:
                continue
            # Get uncovered t-subsets in this k-subset
            tset = ksub_tsets[i]
            gain = 0.0
            for idx in tset:
                if (uncovered_mask >> idx) & 1:
                    gain += w[idx]
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or best_gain <= 0:
            break

        # Add block
        blocks.append(list(all_ksubs[best_idx]))
        block_indices.append(best_idx)
        used[best_idx] = 1

        # Update uncovered mask
        uncovered_mask &= ~ksub_masks[best_idx]
        remaining -= ksub_masks[best_idx].bit_count() if hasattr(ksub_masks[best_idx], 'bit_count') else bin(ksub_masks[best_idx]).count('1')
        # Recalculate remaining properly
        remaining = uncovered_mask.bit_count() if hasattr(uncovered_mask, 'bit_count') else bin(uncovered_mask).count('1')

        # Update dual weights via subgradient descent
        step = 2.0 / (rnd + 1.0)
        # For uncovered t-subsets, increase weight; for covered, decrease
        for idx in range(n_t):
            if (uncovered_mask >> idx) & 1:
                w[idx] += step
            else:
                w[idx] -= step * 0.5
        # Clip to positive
        w = np.maximum(w, 0.1)
        # Renormalize
        w_mean = w.mean()
        if w_mean > 0:
            w /= w_mean

    # --- Minimalization: remove any block whose removal leaves coverage intact ---
    changed = True
    while changed:
        if time.time() > deadline - 15.0:
            break
        changed = False
        new_blocks = []
        new_block_indices = []
        new_used = np.zeros(n_k, dtype=np.int8)

        for i, blk_idx in enumerate(block_indices):
            # Temporarily remove this block from coverage
            # Recompute coverage without this block
            test_mask = 0
            for j, other_idx in enumerate(block_indices):
                if j != i:
                    test_mask |= ksub_masks[other_idx]
            
            # Check if all t-subsets are still covered
            if test_mask == ((1 << n_t) - 1):
                # Safe to drop
                changed = True
            else:
                # Keep it
                new_blocks.append(blocks[i])
                new_block_indices.append(blk_idx)
                new_used[blk_idx] = 1

        blocks = new_blocks
        block_indices = new_block_indices
        used = new_used

    # --- 1-for-1 swap local search to escape local minima ---
    if time.time() < deadline - 10.0:
        n_blocks = len(blocks)
        improved = True
        max_swap_iters = 50
        for _ in range(max_swap_iters):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            # Try each block for a swap with an unused block
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    break
                # Remove block i, see which t-subsets become uncovered
                temp_mask = 0
                for j in range(len(blocks)):
                    if j != i:
                        temp_mask |= ksub_masks[block_indices[j]]
                
                # Find uncovered t-subsets after removal
                uncovered_after_removal = ((1 << n_t) - 1) & ~temp_mask
                if uncovered_after_removal == 0:
                    # Block i is redundant (should have been removed in minimalization)
                    continue
                
                # Find an unused block that covers the most of these uncovered t-subsets
                best_swap_idx = -1
                best_swap_gain = 0
                for j in range(n_k):
                    if used[j]:
                        continue
                    # How many of the uncovered t-subsets does block j cover?
                    common = uncovered_after_removal & ksub_masks[j]
                    gain = common.bit_count() if hasattr(common, 'bit_count') else bin(common).count('1')
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_idx = j
                
                # Only swap if it doesn't increase total blocks (it's 1-for-1)
                # and if it helps (covers at least one new t-subset that wasn't covered by block i)
                if best_swap_idx != -1 and best_swap_gain > 0:
                    # Perform the swap
                    blocks[i] = list(all_ksubs[best_swap_idx])
                    block_indices[i] = best_swap_idx
                    used[block_indices[i]] = 1
                    # Clear the old block's used flag
                    # We need to track which block was replaced
                    # For simplicity, rebuild used
                    improved = True
                    break
            
            if improved:
                # Rebuild used array
                used = np.zeros(n_k, dtype=np.int8)
                for blk_idx in block_indices:
                    used[blk_idx] = 1

    return blocks
# EVOLVE-BLOCK-END
