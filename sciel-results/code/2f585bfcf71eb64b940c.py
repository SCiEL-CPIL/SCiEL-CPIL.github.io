import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145  # Safety margin: evaluator kills at 150s

    # 1. Precompute all t-subsets and k-blocks
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)

    # Build coverage matrix: n_k_blocks x n_t_subs
    coverage_matrix = np.zeros((n_k_blocks, n_t_subs), dtype=np.uint8)
    for i, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            coverage_matrix[i, idx] = 1

    # Precompute for each block the list of t-subset indices it covers
    block_t_sub_indices = []
    for i in range(n_k_blocks):
        block_t_sub_indices.append(np.nonzero(coverage_matrix[i])[0])

    # 2. Greedy construction
    def greedy_construct(seed_order=None, use_randomization=False):
        if time.time() > deadline - 10:
            return None

        if seed_order is not None:
            block_order = list(seed_order)
        else:
            block_order = list(range(n_k_blocks))
            if use_randomization:
                random.shuffle(block_order)

        uncovered = np.ones(n_t_subs, dtype=np.uint8)
        selected = []
        selected_set = set()

        while uncovered.any():
            if time.time() > deadline - 10:
                break

            # Compute gains for all blocks
            gains = (coverage_matrix @ uncovered).astype(np.int32)

            best_gain = 0
            best_idx = -1
            best_candidates = []

            for bi in block_order:
                if bi in selected_set:
                    continue
                g = gains[bi]
                if g > best_gain:
                    best_gain = g
                    best_idx = bi
                    best_candidates = [bi]
                elif g == best_gain and g > 0:
                    best_candidates.append(bi)

            if best_idx == -1 or best_gain == 0:
                break

            if use_randomization and len(best_candidates) > 1:
                best_idx = random.choice(best_candidates)

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered = (uncovered & ~coverage_matrix[best_idx]).astype(np.uint8)

        return selected

    # 3. Redundancy removal
    def remove_redundant(selected, deadline_margin=10):
        if not selected:
            return selected

        current = list(selected)
        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in current:
            cover_count += coverage_matrix[bi]

        improved = True
        while improved and time.time() < deadline - deadline_margin:
            improved = False
            new_current = []
            for i in range(len(current)):
                bi = current[i]
                # Check if removing bi leaves any t-sub with count 0
                if (cover_count[coverage_matrix[bi] == 1] == 1).any():
                    new_current.append(bi)
                else:
                    cover_count -= coverage_matrix[bi]
                    improved = True
            current = new_current

        return current

    # 4. Two-exchange local search: try swapping one block for another
    def local_search_two_exchange(selected, deadline_margin=10):
        if not selected:
            return selected

        current = list(selected)
        current_set = set(current)

        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in current:
            cover_count += coverage_matrix[bi]

        improved = True
        while improved and time.time() < deadline - deadline_margin:
            improved = False
            n = len(current)

            for i in range(n):
                if time.time() > deadline - deadline_margin:
                    break
                bi = current[i]

                # Try removing bi
                cover_count -= coverage_matrix[bi]
                uncovered = (cover_count == 0).astype(np.uint8)

                if not uncovered.any():
                    # Can remove bi entirely
                    current.pop(i)
                    current_set.discard(bi)
                    improved = True
                    break
                else:
                    n_uncovered = int(uncovered.sum())
                    best_gain = 0
                    best_block = -1

                    # Find best replacement
                    for bi2 in range(n_k_blocks):
                        if bi2 in current_set:
                            continue
                        gain = int((coverage_matrix[bi2] & uncovered).sum())
                        if gain > best_gain:
                            best_gain = gain
                            best_block = bi2
                        if best_gain >= n_uncovered:
                            break

                    if best_block != -1 and best_gain >= n_uncovered:
                        # Replace bi with best_block
                        current[i] = best_block
                        current_set.discard(bi)
                        current_set.add(best_block)
                        cover_count += coverage_matrix[best_block]
                        improved = True
                        break
                    else:
                        # Revert
                        cover_count += coverage_matrix[bi]

        return current

    # 5. 2-opt style: try replacing two blocks with one or two different blocks
    def local_search_2opt(selected, deadline_margin=10):
        if not selected or len(selected) < 2:
            return selected

        current = list(selected)
        current_set = set(current)

        cover_count = np.zeros(n_t_subs, dtype=np.int32)
        for bi in current:
            cover_count += coverage_matrix[bi]

        improved = True
        while improved and time.time() < deadline - deadline_margin:
            improved = False
            n = len(current)

            for i in range(n):
                if time.time() > deadline - deadline_margin:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - deadline_margin:
                        break
                    bi = current[i]
                    bj = current[j]

                    # Try removing both
                    cover_count -= coverage_matrix[bi]
                    cover_count -= coverage_matrix[bj]
                    uncovered = (cover_count == 0).astype(np.uint8)

                    if not uncovered.any():
                        # Can remove both
                        current.pop(j)
                        current.pop(i)
                        current_set.discard(bi)
                        current_set.discard(bj)
                        improved = True
                        break
                    else:
                        n_uncovered = int(uncovered.sum())

                        # Try to find one block that covers all uncovered
                        best_gain_one = 0
                        best_block_one = -1
                        for bi2 in range(n_k_blocks):
                            if bi2 in current_set:
                                continue
                            gain = int((coverage_matrix[bi2] & uncovered).sum())
                            if gain > best_gain_one:
                                best_gain_one = gain
                                best_block_one = bi2
                            if best_gain_one >= n_uncovered:
                                break

                        if best_block_one != -1 and best_gain_one >= n_uncovered:
                            # Replace both with one block
                            current.pop(j)
                            current[i] = best_block_one
                            current_set.discard(bi)
                            current_set.discard(bj)
                            current_set.add(best_block_one)
                            cover_count += coverage_matrix[best_block_one]
                            improved = True
                            break
                        else:
                            # Try to find two blocks that cover all uncovered
                            # This is expensive, so only try a limited search
                            best_gain_pair = 0
                            best_pair = None

                            # Use a greedy approach to find two blocks
                            # First, find the best single block
                            if best_block_one != -1:
                                remaining = uncovered.copy()
                                remaining[coverage_matrix[best_block_one] == 1] = 0
                                if not remaining.any():
                                    best_pair = (best_block_one, -1)
                                else:
                                    n_remaining = int(remaining.sum())
                                    best_gain_second = 0
                                    best_block_second = -1
                                    for bi3 in range(n_k_blocks):
                                        if bi3 in current_set or bi3 == best_block_one:
                                            continue
                                        gain = int((coverage_matrix[bi3] & remaining).sum())
                                        if gain > best_gain_second:
                                            best_gain_second = gain
                                            best_block_second = bi3
                                        if best_gain_second >= n_remaining:
                                            break
                                    if best_block_second != -1 and best_gain_second >= n_remaining:
                                        best_pair = (best_block_one, best_block_second)

                            if best_pair is not None:
                                b1, b2 = best_pair
                                current.pop(j)
                                current.pop(i)
                                current.append(b1)
                                if b2 != -1:
                                    current.append(b2)
                                current_set.discard(bi)
                                current_set.discard(bj)
                                current_set.add(b1)
                                if b2 != -1:
                                    current_set.add(b2)
                                cover_count += coverage_matrix[b1]
                                if b2 != -1:
                                    cover_count += coverage_matrix[b2]
                                improved = True
                                break
                            else:
                                # Revert
                                cover_count += coverage_matrix[bi]
                                cover_count += coverage_matrix[bj]

            if improved:
                break  # Restart the loop since indices changed

        return current

    # 6. Multiple restarts with different strategies
    best_blocks = None
    best_count = float('inf')

    np.random.seed(42)
    random.seed(42)

    max_restarts = 15
    time_per_restart = (deadline - time.time() - 15) / max_restarts

    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break

        t_start = time.time()

        # Create a random ordering for greedy
        order = list(range(n_k_blocks))
        random.shuffle(order)

        # Greedy with this ordering
        selected = greedy_construct(seed_order=order, use_randomization=True)

        if selected is None:
            continue

        # Verify coverage
        cov = np.zeros(n_t_subs, dtype=np.uint8)
        for bi in selected:
            cov |= coverage_matrix[bi]
        if not cov.all():
            continue

        # Apply local search improvements
        selected = remove_redundant(selected, deadline_margin=15)
        selected = local_search_two_exchange(selected, deadline_margin=15)
        selected = local_search_2opt(selected, deadline_margin=15)
        selected = remove_redundant(selected, deadline_margin=10)

        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)

        # If we've hit or beaten the known best, try harder
        if best_count <= 151:
            selected = local_search_two_exchange(selected, deadline_margin=10)
            selected = local_search_2opt(selected, deadline_margin=10)
            selected = remove_redundant(selected, deadline_margin=5)
            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = list(selected)

        # Check time
        if time.time() - t_start > time_per_restart * 1.5:
            max_restarts = restart + 3

    # Fallback: simple greedy if nothing found
    if best_blocks is None:
        selected = greedy_construct(use_randomization=False)
        if selected is None:
            # Very basic fallback
            uncovered = set(all_t_subs)
            selected = []
            for bi, block in enumerate(all_k_blocks):
                if not uncovered:
                    break
                if time.time() > deadline - 5:
                    break
                block_t_subs = set(itertools.combinations(block, t))
                if uncovered & block_t_subs:
                    selected.append(bi)
                    uncovered -= block_t_subs
        best_blocks = selected

    return [list(all_k_blocks[i]) for i in best_blocks]
