import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(indices)

    # Reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Build a numpy array: block_to_tsubs_np[i] = array of t-subset indices for block i
    block_to_tsubs_np = np.array(block_to_tsubs, dtype=np.int32)

    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    blocks_in_cover = []
    in_cover_set = set()

    def add_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] += 1
        blocks_in_cover.append(b_idx)
        in_cover_set.add(b_idx)

    def remove_block(b_idx):
        cover_count[block_to_tsubs_np[b_idx]] -= 1
        in_cover_set.discard(b_idx)
        if b_idx in blocks_in_cover:
            blocks_in_cover.remove(b_idx)

    # 1. Greedy Initialization - vectorized gain computation
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)

    while np.any(uncovered_mask):
        if time.time() > deadline - 15:
            break

        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        # Pick a random uncovered t-subset
        target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        candidates = t_sub_to_blocks[target_idx]

        # Vectorized gain computation for all candidates
        cand_arr = np.array(candidates, dtype=np.int32)
        # For each candidate block, count how many of its t-subsets are currently uncovered
        gains = np.sum(cover_count[block_to_tsubs_np[cand_arr]] == 0, axis=1)
        best_local = np.argmax(gains)
        best_block_idx = cand_arr[best_local]
        max_gain = gains[best_local]

        if max_gain > 0:
            add_block(best_block_idx)
            # Update uncovered mask
            newly_covered = block_to_tsubs_np[best_block_idx]
            uncovered_mask[newly_covered] = (cover_count[newly_covered] == 0)

    # 2. Local Search - aggressive deletion and swap
    no_improvement_count = 0
    max_no_improvement = 100

    while time.time() < deadline - 2:
        improved = False

        # Phase 1: Delete all redundant blocks at once
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)

        for b_idx in to_remove:
            remove_block(b_idx)
            improved = True

        if improved:
            no_improvement_count = 0
            continue

        # Phase 2: Try swaps for critical blocks
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if np.any(cover_count[block_to_tsubs_np[b_idx]] == 1):
                critical_blocks.append(b_idx)

        if not critical_blocks:
            break

        # Shuffle critical blocks to avoid bias
        random.shuffle(critical_blocks)

        swapped = False
        for b_old_idx in critical_blocks:
            if time.time() > deadline - 2:
                break

            unique_tsubs = block_to_tsubs_np[b_old_idx][cover_count[block_to_tsubs_np[b_old_idx]] == 1]
            if len(unique_tsubs) == 0:
                continue

            # Find blocks that cover all unique t-subsets
            # Use set intersection
            candidate_set = set(t_sub_to_blocks[unique_tsubs[0]])
            for t_idx in unique_tsubs[1:]:
                candidate_set.intersection_update(t_sub_to_blocks[t_idx])
                if not candidate_set:
                    break

            if candidate_set:
                candidate_set.discard(b_old_idx)
                if candidate_set:
                    # Pick a candidate that maximizes additional coverage
                    best_new = None
                    best_score = -1
                    for b_new_idx in list(candidate_set)[:20]:  # Limit to 20 for speed
                        score = np.sum(cover_count[block_to_tsubs_np[b_new_idx]] == 0)
                        if score > best_score:
                            best_score = score
                            best_new = b_new_idx

                    if best_new is not None:
                        remove_block(b_old_idx)
                        add_block(best_new)
                        improved = True
                        swapped = True
                        break

        if swapped:
            no_improvement_count = 0
            continue

        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            break

    # Final cleanup: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        to_remove = []
        for b_idx in blocks_in_cover:
            if np.all(cover_count[block_to_tsubs_np[b_idx]] > 1):
                to_remove.append(b_idx)
        for b_idx in to_remove:
            remove_block(b_idx)
            changed = True

    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    final_blocks.sort()

    return final_blocks
