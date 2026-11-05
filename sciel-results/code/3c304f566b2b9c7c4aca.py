"""Covering design C(24,7,3): minimize number of 7-subsets of 24 points
so that every 3-subset is contained in some block.
Entry point: construct(v, k, t) -> list of blocks (sorted k-subsets)."""

import itertools
import time
import numpy as np
from math import comb


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # block_triple_indices[i, j] = index of j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    # triple_to_blocks: for each triple, which blocks contain it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for j in range(k_t):
            t_idx = block_triple_indices[i, j]
            triple_to_blocks[t_idx].append(i)

    def _greedy_cover(random_seed):
        """Vectorized greedy: repeatedly pick the block covering the most
        still-uncovered triples, with random tie-breaking."""
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        selected_mask = np.zeros(n_blocks, dtype=bool)

        while remaining > 0:
            # Compute gains for all unselected blocks
            # uncovered_mask[block_triple_indices] -> (n_blocks, k_t) bool
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            gains[selected_mask] = -1

            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]

            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))

            # Update: mark newly covered triples
            b_triples = block_triple_indices[best_idx]
            newly = b_triples[uncovered_mask[b_triples]]
            if len(newly) > 0:
                uncovered_mask[newly] = False
                remaining -= len(newly)

        return selected_blocks

    def _local_improve(blocks, random_seed=0, max_rounds=30):
        """Remove redundant blocks and attempt targeted rebuilds to reduce count."""
        if not blocks:
            return blocks

        # Build coverage counts
        all_triples_set = list(itertools.combinations(range(v), t))
        triple_to_idx_local = {trip: i for i, trip in enumerate(all_triples_set)}
        n_t = len(all_triples_set)

        cov = np.zeros(n_t, dtype=np.int32)
        block_triple_sets = []
        for b in blocks:
            trs = list(itertools.combinations(b, t))
            t_indices = np.array([triple_to_idx_local[tr] for tr in trs], dtype=np.int32)
            block_triple_sets.append(t_indices)
            cov[t_indices] += 1

        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline:
                break

            # 1) Drop redundant block (fast vectorized check)
            n_blocks_now = len(block_triple_sets)
            if n_blocks_now > 0:
                # For each block, count how many of its triples are uniquely covered
                # (cov == 1). If zero, the block is redundant.
                unique_counts = np.array(
                    [np.sum(cov[bt] == 1) for bt in block_triple_sets], dtype=np.int32
                )
                redundant = np.where(unique_counts == 0)[0]
                if len(redundant) > 0:
                    i = int(redundant[0])
                    cov[block_triple_sets[i]] -= 1
                    block_triple_sets.pop(i)
                    blocks.pop(i)
                    improved = True
                    continue

            # 2) Vectorized block swap: remove one block, find best replacement
            if len(blocks) > 1:
                # Find block whose removal leaves the most uncovered triples
                n_blocks_now = len(block_triple_sets)
                if n_blocks_now > 1:
                    # criticality[i] = number of triples uniquely covered by block i
                    criticality = np.array(
                        [np.sum(cov[bt] == 1) for bt in block_triple_sets], dtype=np.int32
                    )
                    best_i = int(np.argmax(criticality))
                    best_uncovered = int(criticality[best_i])

                    if best_uncovered > 0:
                        old_block = blocks[best_i]
                        old_triples = block_triple_sets[best_i].copy()
                        cov[old_triples] -= 1

                        # Find uncovered triples
                        uncovered_mask = (cov == 0)
                        n_uncovered = int(np.sum(uncovered_mask))

                        if n_uncovered > 0:
                            # Vectorized: find the all_block that covers the most
                            # currently-uncovered triples
                            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)

                            # Prefer blocks not already in the solution
                            in_solution = np.zeros(n_blocks, dtype=bool)
                            for bt in block_triple_sets:
                                # We need to map back to block index; use a set of block tuples
                                pass

                            # Simpler: just pick the best block overall, check if it's new
                            best_candidate = int(np.argmax(gains))
                            best_gain = int(gains[best_candidate])

                            # Check if this block is already in our solution
                            new_block_tuple = tuple(all_blocks[best_candidate])
                            existing_tuples = set(tuple(b) for b in blocks)

                            if new_block_tuple not in existing_tuples and best_gain >= best_uncovered:
                                new_block = list(all_blocks[best_candidate])
                                new_triples = block_triple_indices[best_candidate].copy()
                                blocks[best_i] = new_block
                                block_triple_sets[best_i] = new_triples
                                cov[new_triples] += 1
                                improved = True
                            else:
                                # Revert
                                cov[old_triples] += 1

        return blocks

    def _is_valid(blocks):
        """Check every t-subset is covered."""
        covered = set()
        for b in blocks:
            for tr in itertools.combinations(b, t):
                covered.add(tr)
        return len(covered) == n_triples

    # 2. Run multiple greedy restarts
    best_blocks = None
    best_len = float('inf')
    n_restarts = 30  # vectorized greedy is fast; ~1-2s each

    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        blocks = _greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # 3. Local improvement on the best
    if best_blocks is not None:
        improved = _local_improve(best_blocks, random_seed=42, max_rounds=60)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(best_blocks)

    # 4. Additional restarts + local improvement if time remains
    attempt = n_restarts
    while time.time() < deadline - 5.0:
        blocks = _greedy_cover(attempt)
        if len(blocks) < best_len:
            blocks = _local_improve(blocks, random_seed=attempt, max_rounds=40)
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
        attempt += 1
        if attempt > 120:
            break

    # 5. Safety: validate and deduplicate
    if best_blocks is None or not _is_valid(best_blocks):
        # Fallback: single deterministic greedy
        best_blocks = _greedy_cover(999)

    seen = set()
    out = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
