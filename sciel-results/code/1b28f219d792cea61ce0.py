import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)

    # Map each t-subset to an index
    t_sub_to_idx = {}
    for i, sub in enumerate(all_t_subsets):
        t_sub_to_idx[sub] = i

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Build incidence matrix: cover_matrix[i, j] = 1 if block i covers t-subset j
    # For v=19, k=7, t=4: n_k = C(19,7) = 50388, n_t = C(19,4) = 3876
    # This is a 50388 x 3876 boolean matrix - manageable with numpy
    cover_matrix = np.zeros((n_k, n_t), dtype=np.uint8)

    for i, block in enumerate(all_k_subsets):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx[t_sub]
            cover_matrix[i, j] = 1

    # For each t-subset, list of blocks that cover it
    t_sub_to_blocks = [[] for _ in range(n_t)]
    for j in range(n_t):
        col = cover_matrix[:, j]
        t_sub_to_blocks[j] = np.where(col == 1)[0].tolist()

    best_blocks = None
    best_count = float('inf')

    # Multiple restarts with stochastic greedy
    while time.time() < deadline:
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 5.0:
                break

            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]

            # Get candidate blocks that cover this target
            candidates = t_sub_to_blocks[target_idx]
            if not candidates:
                break

            # Compute gains for all candidates using vectorized numpy
            cand_array = np.array(candidates)
            # gains[i] = number of currently uncovered t-subsets covered by block cand_array[i]
            gains = cover_matrix[cand_array, :].dot(uncovered.astype(np.uint8))

            # Stochastic selection: use softmax over gains to allow exploration
            # Higher gains get higher probability, but sub-optimal choices are possible
            if len(gains) > 0 and gains.max() > 0:
                # Temperature-based softmax: lower temp = more greedy
                temp = max(1.0, gains.max() * 0.3)
                log_probs = (gains - gains.max()) / temp
                exp_probs = np.exp(log_probs)
                probs = exp_probs / exp_probs.sum()

                # Sample a block according to these probabilities
                chosen_local = np.random.choice(len(candidates), p=probs)
                chosen_idx = candidates[chosen_local]
            else:
                chosen_idx = candidates[0]

            blocks.append(all_k_subsets[chosen_idx])

            # Update uncovered
            uncovered[cover_matrix[chosen_idx] == 1] = False

        if uncovered.any():
            # Incomplete covering in this restart
            continue

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

            # Local search: try to improve by swapping blocks
            if time.time() < deadline - 10.0:
                best_blocks = local_search_refine(best_blocks, all_k_subsets, cover_matrix, 
                                                   t_sub_to_idx, all_t_subsets, 
                                                   deadline - 10.0)
                if best_blocks is not None:
                    best_count = len(best_blocks)

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[0]
            candidates = t_sub_to_blocks[target_idx]
            if not candidates:
                break

            cand_array = np.array(candidates)
            gains = cover_matrix[cand_array, :].dot(uncovered.astype(np.uint8))
            best_local = np.argmax(gains)
            chosen_idx = candidates[best_local]

            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx] == 1] = False

        return [list(b) for b in blocks]

    return best_blocks


def local_search_refine(blocks, all_k_subsets, cover_matrix, t_sub_to_idx, 
                         all_t_subsets, deadline):
    """Try to reduce the number of blocks by finding redundant blocks or better swaps."""
    n_t = len(all_t_subsets)
    current_blocks = [list(b) for b in blocks]
    n_blocks = len(current_blocks)

    # Map from block tuple to index in all_k_subsets
    block_to_idx = {}
    for i, block in enumerate(all_k_subsets):
        block_to_idx[tuple(block)] = i

    # Compute coverage count for each t-subset
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for block in current_blocks:
        bidx = block_to_idx[tuple(block)]
        coverage_count += cover_matrix[bidx]

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Try removing each block if it's redundant
        for i in range(len(current_blocks)):
            if time.time() > deadline:
                break

            block = current_blocks[i]
            bidx = block_to_idx[tuple(block)]

            # Check if removing this block leaves all t-subsets covered
            new_coverage = coverage_count - cover_matrix[bidx]
            if new_coverage.min() >= 1:
                # Block is redundant - remove it
                coverage_count = new_coverage
                current_blocks.pop(i)
                improved = True
                break  # Restart the loop since indices changed

        # If no removal, try swapping a block with a different one
        if not improved:
            for i in range(len(current_blocks)):
                if time.time() > deadline:
                    break

                block_i = current_blocks[i]
                bidx_i = block_to_idx[tuple(block_i)]

                # Find blocks that could replace this one
                # A replacement block must cover all t-subsets that are only covered by block_i
                only_covered_by_i = (coverage_count == 1) & (cover_matrix[bidx_i] == 1)
                if not only_covered_by_i.any():
                    # Block_i is not the only cover for any t-subset, try to find a better swap
                    # This is complex; skip for now
                    continue

                # Try random swaps: replace block_i with a random other block
                # and check if coverage is maintained
                for _ in range(20):  # Limited attempts
                    if time.time() > deadline:
                        break

                    new_bidx = random.randint(0, len(all_k_subsets) - 1)
                    if new_bidx == bidx_i:
                        continue

                    new_block = all_k_subsets[new_bidx]
                    if new_block in current_blocks:
                        continue

                    # Check if replacement maintains coverage
                    # Remove block_i, add new_block
                    temp_coverage = coverage_count - cover_matrix[bidx_i] + cover_matrix[new_bidx]
                    if temp_coverage.min() >= 1:
                        # Valid swap, but we need to see if it helps
                        # For now, just accept if it doesn't break coverage
                        # (future improvement: check if it enables further reductions)
                        coverage_count = temp_coverage
                        current_blocks[i] = new_block
                        improved = True
                        break

                if improved:
                    break

    return current_blocks
