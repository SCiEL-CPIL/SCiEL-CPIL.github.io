import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
        cols = A.indices[row_start:row_end]
        block_cols.append(cols)

    # Build bitset representation for fast coverage checks
    n_words = (n_t + 63) // 64
    block_bitsets = np.zeros((n_k, n_words), dtype=np.uint64)

    for i, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                word_idx = j // 64
                bit_idx = j % 64
                block_bitsets[i, word_idx] |= (1 << bit_idx)

    def greedy_cover(seed_blocks=None, max_iter=None):
        """Greedy set cover with optional seed blocks."""
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
        """Remove redundant blocks from the covering."""
        if not selected_blocks:
            return selected_blocks

        # Pre-compute lists of t-subset indices for each selected block
        selected_t_subs = []
        for idx in selected_blocks:
            cols = block_cols[idx]
            selected_t_subs.append(cols)

        # Initialize cover_count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for cols in selected_t_subs:
            cover_count[cols] += 1

        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                cols = selected_t_subs[i]
                # Check if any col in cols has cover_count == 1
                is_essential = False
                for c in cols:
                    if cover_count[c] == 1:
                        is_essential = True
                        break

                if not is_essential:
                    to_remove.append(i)

            if to_remove:
                changed = True
                # Remove blocks from the end to maintain indices
                for i in sorted(to_remove, reverse=True):
                    # Update cover_count
                    cols = selected_t_subs[i]
                    cover_count[cols] -= 1
                    # Remove from lists
                    selected_blocks.pop(i)
                    selected_t_subs.pop(i)

        return selected_blocks

    def local_search_improve(selected_blocks, max_rounds=50):
        """Local search with 1-for-1 swaps and deletions, plus 2-for-1 swaps."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)

        def compute_coverage(blocks):
            cov = np.zeros(n_words, dtype=np.uint64)
            for idx in blocks:
                cov |= block_bitsets[idx]
            return cov

        # Initial coverage
        coverage = compute_coverage(best)

        # Count coverage multiplicity
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in best:
            cover_count[block_cols[idx]] += 1

        for rnd in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False

            # Phase 1: Try deletions (if any block is redundant)
            to_delete = []
            for i in range(len(best)):
                cols = block_cols[best[i]]
                redundant = True
                for c in cols:
                    if cover_count[c] == 1:
                        redundant = False
                        break
                if redundant:
                    to_delete.append(i)

            if to_delete:
                # Delete all redundant blocks
                for i in sorted(to_delete, reverse=True):
                    cover_count[block_cols[best[i]]] -= 1
                    best.pop(i)
                best_size = len(best)
                coverage = compute_coverage(best)
                improved = True
                continue

            # Phase 2: Try 1-for-1 swaps
            # For each block i, try replacing with a block that covers more uncovered t-subsets
            for i in range(len(best)):
                if time.time() > deadline:
                    break

                # What would be uncovered if we remove block i?
                others_cov = np.zeros(n_words, dtype=np.uint64)
                for j, idx in enumerate(best):
                    if j != i:
                        others_cov |= block_bitsets[idx]

                would_be_uncovered = coverage & ~others_cov

                if not would_be_uncovered.any():
                    continue  # Block i is redundant, handled in Phase 1

                # Sample candidate replacement blocks
                sample_size = min(200, n_k)
                candidates = random.sample(range(n_k), sample_size)
                candidates = [c for c in candidates if c != best[i]]

                best_candidate = None
                best_score = 0

                for cand in candidates:
                    cand_bits = block_bitsets[cand]

                    # Check if candidate covers all of would_be_uncovered
                    # i.e., (cand_bits & would_be_uncovered) == would_be_uncovered
                    # We need to check bitwise
                    all_covered = True
                    for w in range(n_words):
                        if (cand_bits[w] & would_be_uncovered[w]) != would_be_uncovered[w]:
                            all_covered = False
                            break

                    if not all_covered:
                        continue

                    # Score: number of t-subsets newly covered (not in others_cov)
                    new_cov = cand_bits & ~others_cov
                    score = 0
                    for w in range(n_words):
                        score += bin(new_cov[w]).count('1')

                    if score > best_score:
                        best_score = score
                        best_candidate = cand

                if best_candidate is not None:
                    # Perform the swap
                    cover_count[block_cols[best[i]]] -= 1
                    best[i] = best_candidate
                    cover_count[block_cols[best[i]]] += 1
                    coverage = compute_coverage(best)
                    improved = True

            # Phase 3: Try 2-for-1 swaps (replace one block with two)
            # Only try if we have time
            if time.time() < deadline - 5:
                for i in range(len(best)):
                    if time.time() > deadline - 2:
                        break

                    others_cov = np.zeros(n_words, dtype=np.uint64)
                    for j, idx in enumerate(best):
                        if j != i:
                            others_cov |= block_bitsets[idx]

                    would_be_uncovered = coverage & ~others_cov

                    if not would_be_uncovered.any():
                        continue

                    # Sample pairs of candidate blocks
                    sample_size = min(80, n_k)
                    candidates = random.sample(range(n_k), sample_size)
                    candidates = [c for c in candidates if c != best[i]]

                    best_pair = None
                    best_pair_score = 0

                    for a in range(len(candidates)):
                        for b in range(a + 1, len(candidates)):
                            cand_a = candidates[a]
                            cand_b = candidates[b]
                            pair_bits = block_bitsets[cand_a] | block_bitsets[cand_b]

                            # Check if pair covers all of would_be_uncovered
                            all_covered = True
                            for w in range(n_words):
                                if (pair_bits[w] & would_be_uncovered[w]) != would_be_uncovered[w]:
                                    all_covered = False
                                    break

                            if not all_covered:
                                continue

                            # Score: number of t-subsets newly covered
                            new_cov = pair_bits & ~others_cov
                            score = 0
                            for w in range(n_words):
                                score += bin(new_cov[w]).count('1')

                            if score > best_pair_score:
                                best_pair_score = score
                                best_pair = (cand_a, cand_b)

                    if best_pair is not None:
                        # Replace block i with two blocks
                        cover_count[block_cols[best[i]]] -= 1
                        best.pop(i)
                        best.insert(i, best_pair[0])
                        best.insert(i + 1, best_pair[1])
                        best_size = len(best)
                        cover_count[block_cols[best[i]]] += 1
                        cover_count[block_cols[best[i+1]]] += 1
                        coverage = compute_coverage(best)
                        improved = True

            if not improved:
                break

        return best

    def simulated_annealing(selected_blocks, max_iters=2000):
        """Simulated annealing to escape local minima."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)
        current = list(best)

        # Temperature schedule
        T_start = 3.0
        T_end = 0.1
        n_iters = max_iters

        def compute_coverage(blocks):
            cov = np.zeros(n_words, dtype=np.uint64)
            for idx in blocks:
                cov |= block_bitsets[idx]
            return cov

        current_cov = compute_coverage(current)

        for it in range(n_iters):
            if time.time() > deadline:
                break

            # Temperature decay
            T = T_start * (T_end / T_start) ** (it / n_iters)

            # Generate a random move
            move_type = random.random()
            if move_type < 0.4:
                # Swap: replace one block with another
                i = random.randrange(len(current))
                cand = random.randrange(n_k)
                if cand == current[i]:
                    continue
                new_current = list(current)
                new_current[i] = cand
            elif move_type < 0.7:
                # Delete: remove one block
                if len(current) <= 1:
                    continue
                i = random.randrange(len(current))
                new_current = current[:i] + current[i+1:]
            else:
                # Insert: add one block
                cand = random.randrange(n_k)
                if cand in current:
                    continue
                new_current = current + [cand]

            # Evaluate new solution
            new_cov = compute_coverage(new_current)

            # Check if fully covered
            is_covered = True
            for w in range(n_words):
                if w < n_words - 1:
                    if new_cov[w] != (2**64 - 1):
                        is_covered = False
                        break
                else:
                    # Last word: check only valid bits
                    if n_t % 64 != 0:
                        valid_mask = (1 << (n_t % 64)) - 1
                    else:
                        valid_mask = (2**64 - 1)
                    if (new_cov[w] & valid_mask) != valid_mask:
                        is_covered = False
                        break

            if not is_covered:
                # Accept with probability based on uncovered count
                uncovered_count = 0
                for w in range(n_words):
                    if w < n_words - 1:
                        uncovered_count += 64 - bin(new_cov[w]).count('1')
                    else:
                        if n_t % 64 != 0:
                            valid_mask = (1 << (n_t % 64)) - 1
                            uncovered_count += (n_t % 64) - bin(new_cov[w] & valid_mask).count('1')
                        else:
                            uncovered_count += 64 - bin(new_cov[w]).count('1')

                # Penalty for being uncovered
                delta = uncovered_count * 2.0  # Heavy penalty
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = new_current
                    current_cov = new_cov
            else:
                # Fully covered, compare sizes
                delta = len(new_current) - len(current)
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = new_current
                    current_cov = new_cov
                    if len(current) < best_size:
                        best = list(current)
                        best_size = len(best)

        return best

    # Run greedy with different random seeds
    best_result = None
    best_size = float('inf')

    # First, run standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)

    # Try randomized restarts
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break

        # Create a random seed
        seed_size = min(5, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)

        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)

        if len(result) < best_size:
            best_result = result
            best_size = len(result)

    # Try local search improvement on the best result
    if best_result and time.time() < deadline - 25:
        improved = local_search_improve(best_result, max_rounds=30)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # Try simulated annealing on the best result
    if best_result and time.time() < deadline - 15:
        improved = simulated_annealing(best_result, max_iters=1000)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # Final local search polish
    if best_result and time.time() < deadline - 5:
        improved = local_search_improve(best_result, max_rounds=20)
        improved = remove_redundant(improved)
        if len(improved) < best_size:
            best_result = improved
            best_size = len(improved)

    # Convert indices back to blocks
    if best_result is None:
        best_result = []

    final_blocks = [list(k_subs[i]) for i in best_result]

    return final_blocks
