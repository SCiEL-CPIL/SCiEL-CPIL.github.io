import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (2s before 150s kill)

    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    # Build sparse matrix A of size n_k x n_t
    # A[i, j] = 1 if t_subs[j] is subset of k_subs[i]
    row_indices = []
    col_indices = []
    data = []

    # For each block, find which t-subsets it contains
    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)

    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_k, n_t), dtype=np.int8)

    # Pre-compute the column indices for each block (for fast access)
    block_cols = []
    for i in range(n_k):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end].copy()
        block_cols.append(cols)

    def greedy_cover(seed_blocks=None, max_iter=None, rng=None):
        """Greedy set cover with optional seed blocks."""
        if rng is None:
            rng = random
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []

        # If we have seed blocks, start with them
        if seed_blocks:
            for idx in seed_blocks:
                cols = block_cols[idx]
                u[cols] = 0
                selected_blocks.append(idx)

        if max_iter is None:
            max_iter = 10000

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break

            # Calculate gain for each block
            gains = A @ u
            # Find best block
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]

            if best_gain == 0:
                break

            selected_blocks.append(best_idx)

            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0

        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove redundant blocks from the covering efficiently."""
        if not selected_blocks:
            return selected_blocks

        n_sel = len(selected_blocks)
        # Build cover_count: how many selected blocks cover each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            cols = block_cols[idx]
            cover_count[cols] += 1

        # Track which blocks are still active
        active = [True] * n_sel

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False

            # Find all redundant blocks in one pass
            redundant = []
            for i in range(n_sel):
                if not active[i]:
                    continue
                cols = block_cols[selected_blocks[i]]
                # Check if any col has cover_count == 1 (essential)
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    redundant.append(i)

            if redundant:
                changed = True
                # Remove all redundant blocks
                for i in redundant:
                    cols = block_cols[selected_blocks[i]]
                    cover_count[cols] -= 1
                    active[i] = False

        # Rebuild the list
        return [selected_blocks[i] for i in range(n_sel) if active[i]]

    def local_improve(selected_blocks, max_rounds=5):
        """Improve by trying to remove blocks and adding replacements."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        rng = random

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            # Precompute cover count for all selected blocks
            cover_count = np.zeros(n_t, dtype=np.int16)
            for idx in best:
                cover_count[block_cols[idx]] += 1

            # Try removing each block and finding a replacement
            for i in range(n_best):
                if time.time() > deadline - 2:
                    break

                removed_idx = best[i]
                removed_cols = block_cols[removed_idx]

                # Find t-subsets that would become uncovered
                uncovered_mask = (cover_count[removed_cols] == 1)
                if not uncovered_mask.any():
                    # Block is redundant, just remove it
                    cover_count[removed_cols] -= 1
                    best.pop(i)
                    improved = True
                    break

                uncovered = removed_cols[uncovered_mask]
                n_uncovered = len(uncovered)

                # Find candidate blocks that cover all uncovered t-subsets
                # Use matrix row operations for speed
                best_candidate = -1
                best_score = 0

                # Score all blocks by how many uncovered they cover
                # A[:, uncovered] gives counts for each block
                scores = A[:, uncovered].sum(axis=1)
                
                # Exclude already selected blocks
                for j, idx in enumerate(best):
                    if j != i:
                        scores[idx] = 0
                scores[removed_idx] = 0

                best_candidate = int(np.argmax(scores))
                best_score = scores[best_candidate]

                if best_score >= n_uncovered:
                    # Replace block i with best_candidate
                    cover_count[removed_cols] -= 1
                    best[i] = best_candidate
                    cover_count[block_cols[best_candidate]] += 1
                    improved = True
                    break

            if not improved:
                break

        return best

    def local_improve2(selected_blocks, max_rounds=3):
        """Stronger local search: try 1-for-2 swaps using fast numpy ops."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        rng = random

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            if n_best < 2:
                break

            # Precompute cover count
            cover_count = np.zeros(n_t, dtype=np.int16)
            for idx in best:
                cover_count[block_cols[idx]] += 1

            # Try 1-for-2 swaps: remove 2 blocks, add 1
            max_pairs = min(200, n_best * (n_best - 1) // 2)
            pairs = [(i, j) for i in range(n_best) for j in range(i + 1, n_best)]
            if len(pairs) > max_pairs:
                pairs = rng.sample(pairs, max_pairs)

            for i, j in pairs:
                if time.time() > deadline - 2:
                    break

                idx_i = best[i]
                idx_j = best[j]
                cols_i = block_cols[idx_i]
                cols_j = block_cols[idx_j]

                # Find t-subsets uncovered if we remove both
                # A t-subset is uncovered if cover_count[c] <= 2 and c is in cols_i or cols_j
                # More precisely: uncovered if (c in cols_i and cover_count[c] <= 1 + (c in cols_j)) 
                # or (c in cols_j and cover_count[c] <= 1 + (c in cols_i))
                # Simplify: c is uncovered iff cover_count[c] - (1 if c in cols_i else 0) - (1 if c in cols_j else 0) == 0
                
                # Build uncovered mask
                cc = cover_count.copy()
                cc[cols_i] -= 1
                cc[cols_j] -= 1
                # Uncovered are those with cc == 0 that were covered by i or j
                # Actually, we need: c was covered only by i and/or j
                # c is uncovered iff cc[c] == 0 AND (c in cols_i OR c in cols_j)
                
                # Get union of cols_i and cols_j
                all_cols = np.concatenate([cols_i, cols_j])
                unique_cols = np.unique(all_cols)
                uncovered = unique_cols[cc[unique_cols] == 0]

                if len(uncovered) == 0:
                    # Both are redundant
                    cover_count[cols_i] -= 1
                    cover_count[cols_j] -= 1
                    if j > i:
                        best.pop(j)
                        best.pop(i)
                    else:
                        best.pop(i)
                        best.pop(j)
                    improved = True
                    break

                n_uncovered = len(uncovered)

                # Find best candidate covering all uncovered
                scores = A[:, uncovered].sum(axis=1)
                
                # Exclude already selected blocks
                for m, idx in enumerate(best):
                    if m != i and m != j:
                        scores[idx] = 0

                best_candidate = int(np.argmax(scores))
                best_score = scores[best_candidate]

                if best_score >= n_uncovered:
                    # Replace both with one
                    cover_count[cols_i] -= 1
                    cover_count[cols_j] -= 1
                    new_best = []
                    for m, idx in enumerate(best):
                        if m != i and m != j:
                            new_best.append(idx)
                    new_best.append(best_candidate)
                    cover_count[block_cols[best_candidate]] += 1
                    if len(new_best) < best_size:
                        best = new_best
                        best_size = len(best)
                        improved = True
                        break

            if not improved:
                break

        return best

    # Run greedy with different strategies
    best_result = None
    best_size = float('inf')

    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)

    # Try randomized restarts with different seeds
    num_restarts = 12
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break

        # Create diverse random seeds
        seed_size = min(5, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)

        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)

        if len(result) < best_size:
            best_result = result
            best_size = len(result)

    # Try local improvement on the best result - multiple passes
    if best_result and time.time() < deadline - 20:
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            improved = local_improve(best_result, max_rounds=3)
            improved = remove_redundant(improved)
            if len(improved) < best_size:
                best_result = improved
                best_size = len(improved)
            else:
                break

    # Try stronger local search
    if best_result and time.time() < deadline - 10:
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            improved = local_improve2(best_result, max_rounds=2)
            improved = remove_redundant(improved)
            if len(improved) < best_size:
                best_result = improved
                best_size = len(improved)
            else:
                break

    # Final redundancy removal
    if best_result:
        best_result = remove_redundant(best_result)

    # Convert indices back to blocks
    if best_result is None:
        best_result = []

    final_blocks = [list(k_subs[i]) for i in best_result]

    return final_blocks
