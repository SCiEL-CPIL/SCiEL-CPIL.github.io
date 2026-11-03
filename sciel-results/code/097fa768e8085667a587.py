import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(all_k_subsets)

    # Map block to list of t-subset indices
    block_to_t_indices = []
    for block in all_k_subsets:
        t_indices = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_t_indices.append(t_indices)

    # Map t-subset to list of block indices
    t_to_block_indices = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_t_indices):
        for t_idx in t_indices:
            t_to_block_indices[t_idx].append(b_idx)

    initial_gain = len(block_to_t_indices[0])  # C(k, t)

    # Fast Greedy Initialization with heap
    def run_greedy():
        coverage = np.zeros(num_t_subsets, dtype=np.int8)
        block_gain = np.full(num_k_subsets, initial_gain, dtype=np.int16)

        # Build heap: (-gain, block_idx)
        heap = [(-initial_gain, b) for b in range(num_k_subsets)]
        heapq.heapify(heap)

        selected_blocks = []
        remaining = num_t_subsets

        while remaining > 0:
            # Pop until we find a valid entry
            while heap:
                neg_gain, b_idx = heapq.heappop(heap)
                gain = -neg_gain
                if gain == block_gain[b_idx] and gain > 0:
                    break
            else:
                break  # No more blocks with positive gain

            selected_blocks.append(b_idx)

            # Update coverage and gains
            for t_idx in block_to_t_indices[b_idx]:
                if coverage[t_idx] == 0:
                    coverage[t_idx] = 1
                    remaining -= 1
                    # Update gains for all blocks containing this t_subset
                    for nb_idx in t_to_block_indices[t_idx]:
                        if block_gain[nb_idx] > 0:
                            block_gain[nb_idx] -= 1
                            heapq.heappush(heap, (-block_gain[nb_idx], nb_idx))

        return selected_blocks

    # Run initial greedy
    best_blocks = run_greedy()
    best_size = len(best_blocks)

    # Local Search / Perturbation
    selected = np.zeros(num_k_subsets, dtype=bool)
    for b in best_blocks:
        selected[b] = True

    # Recalculate coverage from scratch for local search state
    coverage_ls = np.zeros(num_t_subsets, dtype=np.int16)
    for b in best_blocks:
        for t_idx in block_to_t_indices[b]:
            coverage_ls[t_idx] += 1

    # Helper to try removing a block
    def try_remove(b_idx):
        nonlocal coverage_ls
        # Check if any t_subset becomes uncovered
        can_remove = True
        for t_idx in block_to_t_indices[b_idx]:
            if coverage_ls[t_idx] == 1:
                can_remove = False
                break
        if can_remove:
            for t_idx in block_to_t_indices[b_idx]:
                coverage_ls[t_idx] -= 1
            selected[b_idx] = False
            return True
        return False

    # Helper to repair (add blocks to cover uncovered sets) using heap
    def repair():
        nonlocal coverage_ls, selected
        # Find uncovered
        uncovered = np.where(coverage_ls == 0)[0]
        if len(uncovered) == 0:
            return True

        # Build heap for repair
        heap = []
        for b_idx in range(num_k_subsets):
            if selected[b_idx]:
                continue
            # Count uncovered t_subsets covered by b_idx
            g = 0
            for t_idx in block_to_t_indices[b_idx]:
                if coverage_ls[t_idx] == 0:
                    g += 1
            if g > 0:
                heapq.heappush(heap, (-g, b_idx))

        if not heap:
            return False

        while uncovered.size > 0:
            # Pop until we find a valid entry
            valid_entry = False
            while heap:
                neg_gain, b_idx = heapq.heappop(heap)
                gain = -neg_gain
                if gain == 0:
                    continue
                # Verify gain is still accurate
                g = 0
                for t_idx in block_to_t_indices[b_idx]:
                    if coverage_ls[t_idx] == 0:
                        g += 1
                if g == gain and g > 0:
                    valid_entry = True
                    break
            if not valid_entry:
                return False

            # Add best_b
            selected[b_idx] = True
            for t_idx in block_to_t_indices[b_idx]:
                if coverage_ls[t_idx] == 0:
                    coverage_ls[t_idx] = 1

            # Update uncovered
            uncovered = np.where(coverage_ls == 0)[0]

            # Update heap with new gains
            for nb_idx in set(t_to_block_indices[t_idx] for t_idx in block_to_t_indices[b_idx] if coverage_ls[t_idx] == 1):
                if not selected[nb_idx]:
                    g = 0
                    for t_idx2 in block_to_t_indices[nb_idx]:
                        if coverage_ls[t_idx2] == 0:
                            g += 1
                    if g > 0:
                        heapq.heappush(heap, (-g, nb_idx))

        return True

    # Final polish: 1-for-1 and 2-for-1 swaps
    def final_polish():
        nonlocal coverage_ls, selected, best_blocks, best_size
        for _ in range(3):  # 3 passes
            # 1-for-1 swaps
            for i, b1 in enumerate(best_blocks):
                if not selected[b1]:
                    continue
                # Try to replace b1 with a better block
                for b2 in range(num_k_subsets):
                    if b2 == b1 or selected[b2]:
                        continue
                    # Check if b2 covers all t_subsets that b1 uniquely covers
                    ok = True
                    for t_idx in block_to_t_indices[b1]:
                        if coverage_ls[t_idx] == 1 and t_idx not in block_to_t_indices[b2]:
                            ok = False
                            break
                    if ok:
                        # Check if b2 adds new coverage
                        adds_new = False
                        for t_idx in block_to_t_indices[b2]:
                            if coverage_ls[t_idx] == 0:
                                adds_new = True
                                break
                        if adds_new:
                            # Perform swap
                            for t_idx in block_to_t_indices[b1]:
                                coverage_ls[t_idx] -= 1
                            for t_idx in block_to_t_indices[b2]:
                                coverage_ls[t_idx] += 1
                            selected[b1] = False
                            selected[b2] = True
                            best_blocks[i] = b2
                            return True
            # 2-for-1 swaps: remove 2 blocks, add 1
            if len(best_blocks) >= 2:
                for i in range(len(best_blocks)):
                    for j in range(i + 1, len(best_blocks)):
                        b1, b2 = best_blocks[i], best_blocks[j]
                        if not selected[b1] or not selected[b2]:
                            continue
                        # Check if we can remove both
                        can_remove_both = True
                        for t_idx in set(block_to_t_indices[b1]) | set(block_to_t_indices[b2]):
                            if coverage_ls[t_idx] <= 2:
                                # Check if t_idx is covered by both or only one
                                in_b1 = t_idx in block_to_t_indices[b1]
                                in_b2 = t_idx in block_to_t_indices[b2]
                                if in_b1 and in_b2:
                                    if coverage_ls[t_idx] == 2:
                                        can_remove_both = False
                                        break
                                elif in_b1 or in_b2:
                                    if coverage_ls[t_idx] == 1:
                                        can_remove_both = False
                                        break
                        if not can_remove_both:
                            continue
                        # Try to find a single block that covers all uncovered t_subsets
                        # This is complex, so we'll skip for now
                        pass
        return False

    # Local Search Loop
    while time.time() < deadline:
        improved = False

        # Phase 1: Try to remove blocks
        blocks_to_check = list(best_blocks)
        random.shuffle(blocks_to_check)
        for b_idx in blocks_to_check:
            if not selected[b_idx]:
                continue
            if try_remove(b_idx):
                best_blocks.remove(b_idx)
                best_size -= 1
                improved = True
                break

        if improved:
            continue

        # Phase 2: Final polish
        if final_polish():
            continue

        # Phase 3: Perturbation
        if best_blocks:
            b_to_remove = random.choice(best_blocks)

            # Save state
            old_coverage = coverage_ls.copy()
            old_selected = selected.copy()
            old_blocks = list(best_blocks)

            # Remove
            for t_idx in block_to_t_indices[b_to_remove]:
                coverage_ls[t_idx] -= 1
            selected[b_to_remove] = False
            best_blocks.remove(b_to_remove)

            # Repair
            if repair():
                new_size = sum(selected)
                if new_size < best_size:
                    best_size = new_size
                    best_blocks = [i for i in range(num_k_subsets) if selected[i]]
                else:
                    # Revert
                    coverage_ls = old_coverage
                    selected = old_selected
                    best_blocks = old_blocks
            else:
                # Revert
                coverage_ls = old_coverage
                selected = old_selected
                best_blocks = old_blocks

    # Return the best found
    return [list(all_k_subsets[b]) for b in best_blocks]
