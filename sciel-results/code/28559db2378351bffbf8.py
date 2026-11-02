"""
Combined approach: greedy with dynamic potential scoring, randomized restarts,
and a working local-search refinement pass.
"""
import time
import random
import numpy as np
from itertools import combinations


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Number of t-subsets per k-subset
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)

    # Build cover matrix: shape (n_k, c_k_t), each entry is an index into all_t_subsets
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]

    best_blocks = None
    best_count = float('inf')

    # Greedy with dynamic potential scoring and random restarts
    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 13)

        uncovered = np.ones(n_t, dtype=bool)
        cover_counts = np.zeros(n_t, dtype=np.int32)
        blocks_found = []

        while True:
            if not np.any(uncovered):
                break

            # Dynamic potential: for each candidate block, sum over its t-subsets
            # of 1 / (cover_count + 1)^2 for uncovered ones, plus a small bonus for
            # covering uncovered triples.
            # We compute gain = number of uncovered triples covered (standard greedy)
            # and potential = sum of 1/(cover_count+1)^2 over the block's triples.
            # Score = alpha * gain + beta * potential
            # But computing potential for all n_k blocks each iteration is expensive.
            # Instead, use a hybrid: standard gain for selection, but break ties
            # using a precomputed "diversity" heuristic.

            gains = np.sum(uncovered[cover_matrix], axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            # Tie-breaking: prefer blocks that cover triples with lower current coverage
            # (i.e., more "rare" triples). Compute a quick score for candidates.
            if len(candidates) > 1:
                # For each candidate, compute the sum of 1/(cover_counts[its triples]+1)
                # This favors blocks that cover under-covered triples.
                cand_scores = np.zeros(len(candidates), dtype=np.float64)
                for ci, ci_idx in enumerate(candidates):
                    triples = cover_matrix[ci_idx]
                    # Only consider uncovered ones
                    mask = uncovered[triples]
                    if np.any(mask):
                        relevant_counts = cover_counts[triples[mask]]
                        cand_scores[ci] = np.sum(1.0 / (relevant_counts + 1.0))
                    else:
                        cand_scores[ci] = 0.0

                # Pick the candidate with highest score; randomize among near-ties
                max_score = np.max(cand_scores)
                near_tie = np.where(cand_scores >= max_score - 1e-9)[0]
                chosen_local = int(near_tie[rng.randint(0, len(near_tie) - 1)])
                chosen_idx = int(candidates[chosen_local])
            else:
                chosen_idx = int(candidates[0])

            blocks_found.append(all_k_subsets[chosen_idx])

            # Update uncovered and cover_counts
            triples = cover_matrix[chosen_idx]
            uncovered[triples] = False
            cover_counts[triples] += 1

        if len(blocks_found) < best_count:
            best_count = len(blocks_found)
            best_blocks = [list(b) for b in blocks_found]

        # If we've hit a good number, try local search
        if best_count <= 78 and time.time() < deadline - 25:
            improved = _local_search(
                best_blocks, v, k, t, cover_matrix, t_sub_index,
                all_k_subsets, all_t_subsets, deadline
            )
            if improved is not None and len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved

    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = []
        uncovered = np.ones(n_t, dtype=bool)
        while np.any(uncovered):
            gains = np.sum(uncovered[cover_matrix], axis=1)
            idx = int(np.argmax(gains))
            if gains[idx] == 0:
                break
            best_blocks.append(list(all_k_subsets[idx]))
            uncovered[cover_matrix[idx]] = False

    return best_blocks


def _local_search(blocks, v, k, t, cover_matrix, t_sub_index,
                  all_k_subsets, all_t_subsets, deadline):
    """Try to reduce the number of blocks by removing one and re-covering."""
    if len(blocks) <= 2:
        return None

    n_t = len(t_sub_index)
    n_k = len(all_k_subsets)

    # Build coverage counts: for each t-subset, how many blocks cover it
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for block in blocks:
        for ts in combinations(block, t):
            cover_counts[t_sub_index[ts]] += 1

    best_result = None
    best_len = len(blocks)

    # Try removing each block and see if we can re-cover with fewer new blocks
    for remove_idx in range(len(blocks)):
        if time.time() > deadline - 5:
            break

        removed_block = blocks[remove_idx]
        removed_triples = list(combinations(removed_block, t))

        # Tentatively remove this block
        for ts in removed_triples:
            cover_counts[t_sub_index[ts]] -= 1

        # Find uncovered t-subsets
        uncovered_mask = cover_counts == 0
        n_uncovered = int(np.sum(uncovered_mask))

        if n_uncovered == 0:
            # Block was redundant
            candidate = blocks[:remove_idx] + blocks[remove_idx + 1:]
            if len(candidate) < best_len:
                best_len = len(candidate)
                best_result = candidate
        else:
            # Try to cover the uncovered t-subsets with one new block
            # Find which candidate block covers the most uncovered t-subsets
            # and does not duplicate an existing block
            existing_set = set(tuple(b) for b in blocks)
            existing_set.discard(tuple(removed_block))

            # Score each candidate block by how many uncovered triples it covers
            # and that it's not already in the set
            gains = np.sum(uncovered_mask[cover_matrix], axis=1)

            # Exclude blocks already in the set
            # This is expensive; instead, just check after selection
            max_gain = np.max(gains)
            if max_gain > 0:
                candidates = np.where(gains == max_gain)[0]
                # Try a few candidates
                for _ in range(min(5, len(candidates))):
                    cand_idx = int(candidates[random.randint(0, len(candidates) - 1)])
                    cand_block = all_k_subsets[cand_idx]
                    if tuple(cand_block) not in existing_set:
                        candidate = blocks[:remove_idx] + blocks[remove_idx + 1:] + [list(cand_block)]
                        # Verify it's a valid cover
                        if _is_valid_cover(candidate, v, k, t, t_sub_index):
                            if len(candidate) < best_len:
                                best_len = len(candidate)
                                best_result = candidate
                        break  # Only try one replacement per removal for speed

        # Restore
        for ts in removed_triples:
            cover_counts[t_sub_index[ts]] += 1

    return best_result


def _is_valid_cover(blocks, v, k, t, t_sub_index):
    """Check if blocks form a valid t-cover of v points."""
    n_t = len(t_sub_index)
    covered = np.zeros(n_t, dtype=bool)
    for block in blocks:
        for ts in combinations(block, t):
            covered[t_sub_index[ts]] = True
    return bool(np.all(covered))
