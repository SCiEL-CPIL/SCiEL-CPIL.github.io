"""Covering design C(19,7,4) via vectorized greedy + local search with redundancy removal."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    ts_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-subsets and their coverage (which t-subsets they cover)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # coverage_matrix[i, j] = t-subset index j covered by k-subset i
    coverage_matrix = np.full((n_ksubs, 2 ** t), -1, dtype=np.int32)
    for i, ksub in enumerate(all_ksubs):
        for j, ts in enumerate(itertools.combinations(ksub, t)):
            coverage_matrix[i, j] = ts_to_idx[ts]

    # Block index lookup: tuple -> index in all_ksubs
    ksub_to_idx = {ksub: i for i, ksub in enumerate(all_ksubs)}

    def get_block_indices(block):
        """Get the index of a block in all_ksubs."""
        return ksub_to_idx[tuple(sorted(block))]

    def greedy_construct():
        """Greedy: repeatedly pick the block that covers the most uncovered t-subsets."""
        uncovered_mask = np.ones(n_tsubsets, dtype=np.int8)
        blocks = []

        while np.any(uncovered_mask):
            if time.time() > deadline - 2:
                break

            # Vectorized: compute marginal gain for each candidate block
            # coverage_matrix has shape (n_ksubs, 16), values are t-subset indices or -1
            # gain[i] = number of t-subsets covered by block i that are still uncovered
            cov = coverage_matrix  # (n_ksubs, 16)
            # For each block, count how many of its t-subsets are uncovered
            # cov values are indices into uncovered_mask, or -1
            # We need: for each row, count entries where uncovered_mask[cov[i,j]] == 1
            # Use np.take with where
            gains = np.zeros(n_ksubs, dtype=np.int32)
            for j in range(2 ** t):
                col = cov[:, j]  # (n_ksubs,)
                valid = col >= 0
                if np.any(valid):
                    gains[valid] += uncovered_mask[col[valid]]

            # Pick the block with max gain
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                # No block can help (shouldn't happen if we're not done)
                break

            best_block = list(all_ksubs[best_idx])
            blocks.append(best_block)

            # Update uncovered mask
            for j in range(2 ** t):
                idx = coverage_matrix[best_idx, j]
                if idx >= 0:
                    uncovered_mask[idx] = 0

        return blocks

    # Initial greedy construction
    blocks = greedy_construct()
    best_blocks = list(blocks)
    best_n = len(blocks)

    def compute_coverage_counts(block_indices):
        """Count how many blocks cover each t-subset. block_indices: list of k-sub indices."""
        if not block_indices:
            return np.zeros(n_tsubsets, dtype=np.int32)
        arr = np.array(block_indices, dtype=np.int64)
        # For each block, it covers 35 t-subsets
        # coverage_matrix has shape (n_ksubs, 16)
        cov = coverage_matrix[arr]  # (n_blocks, 16)
        # Flatten valid entries
        valid = cov >= 0
        flat = cov[valid]
        return np.bincount(flat, minlength=n_tsubsets).astype(np.int32)

    def try_remove_redundant(block_indices):
        """Iteratively remove blocks that have no unique t-subsets."""
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            counts = compute_coverage_counts(block_indices)

            for i in range(len(block_indices)):
                if time.time() > deadline - 2:
                    break
                bi = block_indices[i]
                # Check if any t-subset is uniquely covered by this block
                cov_row = coverage_matrix[bi]
                valid = cov_row >= 0
                if np.any(counts[cov_row[valid]] == 1):
                    continue  # block has a unique t-subset, keep it
                # Remove this block
                del block_indices[i]
                changed = True
                break
        return block_indices

    def is_valid(block_indices):
        """Check if all t-subsets are covered."""
        if not block_indices:
            return n_tsubsets == 0
        counts = compute_coverage_counts(block_indices)
        return np.all(counts > 0)

    def local_search(block_indices):
        """Try to improve by removing redundant blocks and swapping."""
        current = list(block_indices)
        current = try_remove_redundant(current)
        best = list(current)
        best_n = len(current)

        start_time = time.time()
        while time.time() - start_time < 120 and time.time() < deadline - 3:
            if len(current) <= 10:
                break

            # Try random removal + repair
            idx = rng.randint(0, len(current) - 1)
            removed = current[idx]
            current.pop(idx)

            # Check if still valid
            if is_valid(current):
                if len(current) < best_n:
                    best = list(current)
                    best_n = len(current)
                current = try_remove_redundant(current)
                if len(current) < best_n:
                    best = list(current)
                    best_n = len(current)
            else:
                # Find uncovered t-subsets
                counts = compute_coverage_counts(current)
                uncovered_mask = (counts == 0).astype(np.int8)

                if np.any(uncovered_mask):
                    # Add greedy block for uncovered
                    cov = coverage_matrix
                    gains = np.zeros(n_ksubs, dtype=np.int32)
                    for j in range(2 ** t):
                        col = cov[:, j]
                        valid = col >= 0
                        if np.any(valid):
                            gains[valid] += uncovered_mask[col[valid]]

                    best_block_idx = int(np.argmax(gains))
                    if gains[best_block_idx] > 0:
                        current.append(best_block_idx)
                    else:
                        current.append(removed)
                        continue

                    # Try to remove more
                    current = try_remove_redundant(current)
                    if len(current) < best_n:
                        best = list(current)
                        best_n = len(current)

        return best, best_n

    # Convert blocks to indices
    block_indices = [get_block_indices(b) for b in best_blocks]

    # Run local search
    improved_indices, improved_n = local_search(block_indices)
    if improved_n < best_n:
        best_blocks = [list(all_ksubs[i]) for i in improved_indices]
        best_n = improved_n

    # Final redundancy removal
    final_indices = [get_block_indices(b) for b in best_blocks]
    final_indices = try_remove_redundant(final_indices)
    final_blocks = [list(all_ksubs[i]) for i in final_indices]

    # Ensure validity
    if not is_valid(final_indices):
        final_blocks = [list(all_ksubs[i]) for i in block_indices]

    return [sorted(b) for b in final_blocks]
