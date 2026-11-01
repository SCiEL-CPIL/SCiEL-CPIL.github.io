import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute block_to_tsubs
    block_to_tsubs = []
    for B in all_kblocks:
        covered = [tsub_to_idx[ts] for ts in itertools.combinations(B, t)]
        block_to_tsubs.append(covered)

    # Build Cover matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i in range(n_blocks):
        for ts in block_to_tsubs[i]:
            Cover[i, ts] = 1

    # --- Stage 1: Greedy Set Cover ---
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []

    while True:
        if time.time() > deadline - 30.0:
            break
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        if best_gain == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        if np.sum(uncovered) == 0:
            break

    # Fallback if greedy didn't finish
    if np.sum(uncovered) > 0:
        while np.sum(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            gains = Cover @ uncovered
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            uncovered = uncovered & (Cover[best_idx] == 0)

    # --- Stage 2: Optimization ---
    selected_set = set(selected_indices)
    selected_list = list(selected_set)

    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_list:
        cover_counts += Cover[idx]

    # Ensure validity
    if np.any(cover_counts == 0):
        missing = np.where(cover_counts == 0)[0]
        for j in missing:
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                idx = candidates[0]
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_list.append(idx)
                    cover_counts += Cover[idx]

    # Helper: remove redundant blocks, return new list and updated cover_counts
    def remove_redundant(sel_list, cc):
        changed = True
        while changed:
            changed = False
            new_list = []
            for idx in sel_list:
                if np.any((Cover[idx] == 1) & (cc == 1)):
                    new_list.append(idx)
                else:
                    cc -= Cover[idx]
                    changed = True
            sel_list = new_list
        return sel_list, cc

    # Phase 1: Remove redundant blocks
    selected_list, cover_counts = remove_redundant(selected_list, cover_counts)

    # Phase 2: Local search with proper state management
    # Use Add-Minimize and Swap moves

    best_size = len(selected_list)
    best_selected_list = selected_list[:]
    best_cover_counts = cover_counts.copy()

    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break

        improved = False
        current_size = len(selected_list)

        # Shuffle for randomization
        indices_to_try = selected_list[:]
        random.shuffle(indices_to_try)

        for b_idx in indices_to_try:
            if b_idx not in selected_set:
                continue
            if time.time() > deadline - 5.0:
                break

            # Get unique t-subsets for b_idx
            unique_tsubs = [s for s in block_to_tsubs[b_idx] if cover_counts[s] == 1]

            if not unique_tsubs:
                # Redundant, remove it
                cover_counts -= Cover[b_idx]
                selected_set.discard(b_idx)
                selected_list.remove(b_idx)
                improved = True
                continue

            # Try to find a replacement that covers all unique_tsubs
            first_s = unique_tsubs[0]
            candidates = np.where(Cover[:, first_s] == 1)[0]

            valid_candidates = []
            for c in candidates:
                if c in selected_set:
                    continue
                covers_all = all(Cover[c, s] == 1 for s in unique_tsubs)
                if covers_all:
                    valid_candidates.append(c)

            if not valid_candidates:
                continue

            # Pick a random candidate
            c_idx = random.choice(valid_candidates)

            # Tentative swap: work on copies
            test_cc = cover_counts.copy()
            test_list = selected_list[:]
            test_set = selected_set.copy()

            # Remove b_idx
            test_cc -= Cover[b_idx]
            test_set.discard(b_idx)
            test_list.remove(b_idx)

            # Add c_idx
            test_cc += Cover[c_idx]
            test_set.add(c_idx)
            test_list.append(c_idx)

            # Remove redundant blocks from test state
            test_list, test_cc = remove_redundant(test_list, test_cc)

            if len(test_list) < current_size:
                # Keep the improvement
                selected_list = test_list
                selected_set = test_set
                cover_counts = test_cc
                improved = True
                current_size = len(selected_list)
                if current_size < best_size:
                    best_size = current_size
                    best_selected_list = selected_list[:]
                    best_cover_counts = cover_counts.copy()

        # Add-Minimize: try adding a block that enables removal of multiple others
        if time.time() < deadline - 5.0:
            # Find blocks not in solution
            not_in_solution = [i for i in range(n_blocks) if i not in selected_set]
            random.shuffle(not_in_solution)

            for add_idx in not_in_solution[:50]:  # Limit to 50 attempts
                if time.time() > deadline - 5.0:
                    break

                test_cc = cover_counts.copy()
                test_list = selected_list[:]
                test_set = selected_set.copy()

                # Add the block
                test_cc += Cover[add_idx]
                test_set.add(add_idx)
                test_list.append(add_idx)

                # Remove redundant blocks
                test_list, test_cc = remove_redundant(test_list, test_cc)

                if len(test_list) < len(selected_list):
                    selected_list = test_list
                    selected_set = test_set
                    cover_counts = test_cc
                    improved = True
                    if len(selected_list) < best_size:
                        best_size = len(selected_list)
                        best_selected_list = selected_list[:]
                        best_cover_counts = cover_counts.copy()

        if not improved:
            # Try restarting with a perturbation
            if time.time() < deadline - 10.0:
                # Perturb: remove some random blocks and re-greedy the uncovered
                perturbation_size = max(1, len(selected_list) // 10)
                to_remove = random.sample(selected_list, min(perturbation_size, len(selected_list)))
                for idx in to_remove:
                    cover_counts -= Cover[idx]
                    selected_set.discard(idx)
                    selected_list.remove(idx)

                # Re-greedy uncovered
                while True:
                    if time.time() > deadline - 5.0:
                        break
                    uncovered_mask = (cover_counts == 0).astype(np.uint8)
                    if np.sum(uncovered_mask) == 0:
                        break
                    gains = Cover @ uncovered_mask
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    selected_set.add(best_idx)
                    selected_list.append(best_idx)
                    cover_counts += Cover[best_idx]

                # Remove redundant
                selected_list, cover_counts = remove_redundant(selected_list, cover_counts)

    # Restore best solution found
    if len(best_selected_list) < len(selected_list):
        selected_list = best_selected_list
        cover_counts = best_cover_counts

    # Final validity check and repair
    final_blocks = [list(all_kblocks[i]) for i in selected_list]

    final_cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for B in final_blocks:
        for ts in itertools.combinations(B, t):
            final_cover_counts[tsub_to_idx[ts]] += 1

    if np.any(final_cover_counts == 0):
        missing = np.where(final_cover_counts == 0)[0]
        for j in missing:
            candidates = np.where(Cover[:, j] == 1)[0]
            if len(candidates) > 0:
                for c in candidates:
                    if c not in selected_set:
                        final_blocks.append(list(all_kblocks[c]))
                        selected_set.add(c)
                        final_cover_counts += Cover[c]
                        break

    return final_blocks
