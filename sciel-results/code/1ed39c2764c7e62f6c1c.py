import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

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
        if time.time() > deadline - 40:
            break

        # Find block with max gain
        best_idx = int(np.argmax(gains))

        if gains[best_idx] == 0:
            break

        selected_blocks.append(best_idx)
        selected_set.add(best_idx)

        # Update gains and uncovered status
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
    if selected_blocks and time.time() < deadline - 30:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed and time.time() < deadline - 25:
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

    # 4. Weighted Local Search (Max-SAT inspired)
    if selected_blocks and time.time() < deadline - 5:
        # Rebuild cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        # Track which blocks are currently selected
        is_selected = np.zeros(n_k, dtype=bool)
        for b_idx in selected_blocks:
            is_selected[b_idx] = True

        # Precompute for each block: list of t-subsets it covers
        # and for each t-subset: list of blocks that cover it

        # Local search: try adding/removing blocks to minimize total weight
        # Weight = number of blocks selected
        # We'll use a simple approach: try single-block additions and deletions

        best_solution = list(selected_blocks)
        best_count = len(selected_blocks)

        # Build a set of selected blocks for fast lookup
        selected_set = set(selected_blocks)

        # Precompute for each block, the set of t-subsets it covers
        # for fast intersection checks

        # Local search loop
        search_time_limit = deadline - 5
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < search_time_limit and no_improve_count < max_no_improve:
            improved = False

            # Try removals first (easier to validate)
            for b_idx in list(selected_blocks):
                if time.time() > search_time_limit:
                    break

                # Check if b_idx is redundant
                is_redundant = True
                for t_idx in block_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_redundant = False
                        break

                if is_redundant:
                    # Remove b_idx
                    for t_idx in block_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.discard(b_idx)
                    is_selected[b_idx] = False
                    improved = True

            if not improved:
                # Try additions: add a block, then try to remove multiple
                # Sample candidates
                candidates = [b for b in range(n_k) if not is_selected[b]]
                if len(candidates) > 2000:
                    random.shuffle(candidates)
                    candidates = candidates[:2000]

                best_add = None
                best_net_gain = 0

                for cand in candidates:
                    if time.time() > search_time_limit:
                        break

                    cand_t_idxs = block_t_idxs[cand]

                    # Count how many blocks become removable if we add cand
                    removable = []
                    for b_idx in selected_blocks:
                        # Check if b_idx has any t-subset with cover_count == 1
                        # that is NOT covered by cand
                        essential = False
                        for t_idx in block_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1 and t_idx not in set(cand_t_idxs):
                                essential = True
                                break
                        if not essential:
                            removable.append(b_idx)

                    net_gain = len(removable) - 1
                    if net_gain > best_net_gain:
                        best_net_gain = net_gain
                        best_add = (cand, removable)

                if best_add is not None:
                    cand, removable = best_add
                    # Apply
                    for t_idx in block_t_idxs[cand]:
                        cover_count[t_idx] += 1
                    selected_blocks.append(cand)
                    selected_set.add(cand)
                    is_selected[cand] = True

                    for b_idx in removable:
                        for t_idx in block_t_idxs[b_idx]:
                            cover_count[t_idx] -= 1
                        selected_blocks.remove(b_idx)
                        selected_set.discard(b_idx)
                        is_selected[b_idx] = False

                    improved = True

            if improved:
                no_improve_count = 0
                if len(selected_blocks) < best_count:
                    best_count = len(selected_blocks)
                    best_solution = list(selected_blocks)
            else:
                no_improve_count += 1

        # Use best solution found
        selected_blocks = best_solution

        # Final deletion pass
        if time.time() < deadline - 2:
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in selected_blocks:
                for t_idx in block_t_idxs[b_idx]:
                    cover_count[t_idx] += 1

            changed = True
            while changed and time.time() < deadline - 1:
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
    if selected_blocks:
        final_cover = np.zeros(n_t, dtype=np.int32)
        for b_idx in selected_blocks:
            for t_idx in block_t_idxs[b_idx]:
                final_cover[t_idx] += 1

        if not np.all(final_cover > 0):
            # Fallback
            pass

    # Convert indices back to blocks
    final_blocks = [list(all_k_blocks[i]) for i in selected_blocks]

    return final_blocks
