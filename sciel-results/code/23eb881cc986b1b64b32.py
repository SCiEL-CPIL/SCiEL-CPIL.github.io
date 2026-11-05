import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence matrix: blocks x t_subsets
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            rows.append(b_idx)
            cols.append(t_subset_to_idx[t_sub])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    incidence = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                            shape=(num_k_blocks, num_t_subsets))

    # For each t-subset, which blocks cover it
    t_subset_to_blocks = [np.where(incidence[:, i] != 0)[0] for i in range(num_t_subsets)]

    # Precompute block t-subset indices for fast gain computation
    block_t_indices = []
    for b_idx in range(num_k_blocks):
        idxs = np.where(incidence[b_idx] != 0)[0]
        block_t_indices.append(idxs)

    best_blocks = None
    best_score = float('inf')

    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        random.seed(restart)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        block_indices = []
        uncovered_count = num_t_subsets

        # Greedy phase with vectorized gain
        while uncovered_count > 0:
            if time.time() > deadline - 15.0:
                break

            # Find an uncovered t-subset
            unc_mask = (coverage == 0)
            unc_indices = np.where(unc_mask)[0]
            if len(unc_indices) == 0:
                break
            u_idx = unc_indices[random.randint(0, len(unc_indices) - 1)]

            candidates = t_subset_to_blocks[u_idx]

            # Vectorized gain computation
            gains = np.array([np.count_nonzero(coverage[block_t_indices[bi]] == 0)
                              for bi in candidates])
            best_bi = candidates[np.argmax(gains)]

            block_indices.append(best_bi)
            idxs = block_t_indices[best_bi]
            newly = coverage[idxs] == 0
            coverage[idxs] += 1
            uncovered_count -= np.count_nonzero(newly)

        # Check if complete
        if uncovered_count > 0:
            # Repair phase
            while uncovered_count > 0 and time.time() < deadline - 10.0:
                unc_mask = (coverage == 0)
                unc_indices = np.where(unc_mask)[0]
                if len(unc_indices) == 0:
                    break
                u_idx = unc_indices[0]
                candidates = t_subset_to_blocks[u_idx]
                gains = np.array([np.count_nonzero(coverage[block_t_indices[bi]] == 0)
                                  for bi in candidates])
                best_bi = candidates[np.argmax(gains)]
                block_indices.append(best_bi)
                idxs = block_t_indices[best_bi]
                newly = coverage[idxs] == 0
                coverage[idxs] += 1
                uncovered_count -= np.count_nonzero(newly)

        # Reduction phase
        changed = True
        while changed:
            if time.time() > deadline - 8.0:
                break
            changed = False
            for i in range(len(block_indices) - 1, -1, -1):
                if time.time() > deadline - 8.0:
                    break
                bi = block_indices[i]
                idxs = block_t_indices[bi]
                if np.any(coverage[idxs] == 1):
                    continue
                coverage[idxs] -= 1
                block_indices.pop(i)
                changed = True

        if len(block_indices) < best_score:
            best_score = len(block_indices)
            best_blocks = [list(all_k_blocks[bi]) for bi in block_indices]

    return best_blocks
