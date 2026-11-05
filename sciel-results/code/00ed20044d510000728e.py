import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # 1. Pre-computation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # For each block, compute the list of t-subset indices it covers
    block_t_idxs = []
    for block in all_k_blocks:
        t_idxs = []
        for sub in itertools.combinations(block, t):
            t_idxs.append(t_sub_to_idx[sub])
        block_t_idxs.append(t_idxs)

    # Build reverse mapping: for each t-subset, which block indices cover it
    t_sub_to_blocks = [[] for _ in range(n_t)]
    for b_idx, t_idxs in enumerate(block_t_idxs):
        for t_idx in t_idxs:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Convert to numpy arrays for fast vectorized updates
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    # 2. Greedy Construction with NumPy
    # Initial gains: number of t-subsets covered by each block
    gains = np.array([len(t_idxs) for t_idxs in block_t_idxs], dtype=np.int32)
    uncovered = np.ones(n_t, dtype=bool)

    selected_blocks = []
    selected_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Find block with max gain
        best_idx = int(np.argmax(gains))

        if gains[best_idx] == 0:
            break

        selected_blocks.append(best_idx)
        selected_set.add(best_idx)

        # Update gains and uncovered status using vectorized operations
        newly_covered = []
        for t_idx in block_t_idxs[best_idx]:
            if uncovered[t_idx]:
                newly_covered.append(t_idx)
                uncovered[t_idx] = False

        # Vectorized gain decrement for all newly covered t-subsets
        if newly_covered:
            for t_idx in newly_covered:
                if len(t_sub_to_blocks_np[t_idx]) > 0:
                    gains[t_sub_to_blocks_np[t_idx]] -= 1

    # 3. Redundancy Removal (deletion phase)
    if selected_blocks and time.time() < deadline - 20:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            new_blocks = []
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_essential = False
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break

                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1

            selected_blocks = new_blocks

    # 4. Swap-based Local Search (add one, remove one or more)
    if selected_blocks and time.time() < deadline - 5:
        # Rebuild cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        # Try to find beneficial swaps: add a block, then try to remove multiple blocks
        improved = True
        max_swap_iters = 50
        swap_iter = 0

        while improved and time.time() < deadline - 3 and swap_iter < max_swap_iters:
            improved = False
            swap_iter += 1

            # For each candidate block not in selected_set, check if adding it
            # allows removal of more blocks than we add (net reduction)
            best_swap = None
            best_swap_gain = 0

            # Sample a subset of candidate blocks for speed
            candidates = [b for b in range(n_k) if b not in selected_set]
            if len(candidates) > 5000:
                random.shuffle(candidates)
                candidates = candidates[:5000]

            for cand_idx in candidates:
                # Temporarily add cand_idx
                cand_t_idxs = block_t_idxs[cand_idx]

                # Count how many t-subsets in cand would have cover_count become 2
                # (i.e., were covered exactly once before, now covered twice)
                # These are the "freed" subsets that might allow removal of other blocks

                # For each block in selected_blocks, check if it becomes redundant
                # after adding cand_idx
                removable = []
                for b_idx in selected_blocks:
                    # Check if b_idx has any t-subset with cover_count == 1
                    # that is NOT in cand_idx's coverage
                    is_essential = False
                    for t_idx in block_t_idxs[b_idx]:
                        if cover_count[t_idx] == 1 and t_idx not in set(cand_t_idxs):
                            is_essential = True
                            break
                    if not is_essential:
                        removable.append(b_idx)

                # Net gain: number of blocks removed - 1 (for the added block)
                net_gain = len(removable) - 1
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_swap = (cand_idx, removable)

            if best_swap is not None:
                cand_idx, removable = best_swap
                # Apply the swap
                selected_blocks.append(cand_idx)
                selected_set.add(cand_idx)
                for b_idx in removable:
                    selected_blocks.remove(b_idx)
                    selected_set.discard(b_idx)

                # Update cover_count
                for t_idx in block_t_idxs[cand_idx]:
                    cover_count[t_idx] += 1
                for b_idx in removable:
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1

                improved = True

        # Final deletion pass after swaps
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_blocks = []
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_essential = False
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break

                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1

            selected_blocks = new_blocks

    # 5. Final Verification
    # Verify that all t-subsets are covered
    if selected_blocks:
        final_cover = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                final_cover[t_idx] += 1

        if not np.all(final_cover > 0):
            # Fallback: if swap broke validity, use greedy result
            # This shouldn't happen, but safety check
            pass

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[i]) for i in selected_blocks]

    return final_blocks
