"""Covering design C(v,k,t) using numpy-accelerated greedy set cover
with local optimization (redundancy removal and block replacement)."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # Greedy construction using matrix-vector product for gains
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    remaining_time_limit = deadline - 20.0  # Leave time for post-processing

    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break

        # Compute gains: number of uncovered t-subsets each block covers
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))

        if gains[best_idx] == 0:
            break

        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    blocks = [all_kblocks[i] for i in selected_indices]

    # Post-processing: Remove redundant blocks and try local improvements
    if time.time() < deadline - 5.0:
        blocks = _local_optimize(blocks, v, k, t, Cover, all_kblocks,
                                 tsub_to_idx, deadline - 5.0)

    return blocks


def _local_optimize(blocks, v, k, t, Cover, all_kblocks, tsub_to_idx, deadline):
    """Remove redundant blocks and attempt block replacements."""
    start = time.time()
    n_tsubs = len(list(itertools.combinations(range(v), t)))

    # Compute coverage counts for each t-subset
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    block_indices = []
    for B in blocks:
        # Find index of B in all_kblocks
        idx = all_kblocks.index(B)
        block_indices.append(idx)
        cover_counts += Cover[idx]

    # Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Check if this block covers any t-subset with count 1
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_block_indices.append(idx)
            else:
                # Remove this block
                cover_counts -= Cover[idx]
                changed = True
        block_indices = new_block_indices

    # Try block replacement: for each block, see if another block covers
    # all the same "critical" t-subsets (those with count 1)
    if time.time() < deadline - 5.0:
        block_indices = _try_block_replacement(block_indices, Cover,
                                                all_kblocks, cover_counts,
                                                deadline - 5.0)

    return [all_kblocks[i] for i in block_indices]


def _try_block_replacement(block_indices, Cover, all_kblocks, cover_counts, deadline):
    """Try to replace blocks with alternatives that cover the same critical t-subsets."""
    start = time.time()
    n_blocks = len(all_kblocks)

    # For each block, find its critical t-subsets (count == 1)
    # Try to find an alternative block that covers all of them
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_block_indices = []
        for idx in block_indices:
            # Find critical t-subsets covered by this block
            critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
            critical_tsubs = np.where(critical_mask)[0]

            if len(critical_tsubs) == 0:
                # No critical t-subsets; this block is redundant
                cover_counts -= Cover[idx]
                changed = True
                continue

            # Try to find an alternative block that covers all critical t-subsets
            # and doesn't increase the total block count
            best_alt = -1
            best_score = -1
            for alt_idx in range(len(all_kblocks)):
                if alt_idx == idx:
                    continue
                # Check if alt covers all critical t-subsets
                if np.all(Cover[alt_idx][critical_tsubs] == 1):
                    # Score: prefer blocks that cover more uncovered or low-count t-subsets
                    # Simple heuristic: count of t-subsets with count <= 2
                    score = np.sum((Cover[alt_idx] == 1) & (cover_counts <= 2))
                    if score > best_score:
                        best_score = score
                        best_alt = alt_idx

            if best_alt != -1:
                # Replace
                cover_counts -= Cover[idx]
                cover_counts += Cover[best_alt]
                new_block_indices.append(best_alt)
                changed = True
            else:
                new_block_indices.append(idx)

        block_indices = new_block_indices

    return block_indices


if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
