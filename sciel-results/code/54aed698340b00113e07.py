import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)  # number of t-subsets in a k-block

    # Precompute: for each block, which t-subsets it covers
    block_t_indices = []
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])

    # Store as list of arrays (variable length not an issue here, all same length)
    block_t_indices_np = [np.array(lst, dtype=np.int32) for lst in block_t_indices]

    # Build reverse mapping: for each t-subset, which blocks cover it
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx, subs in enumerate(block_t_indices):
        for ts_idx in subs:
            ts_to_blocks[ts_idx].append(b_idx)
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ts_to_blocks]

    # Build a set of frozensets for fast membership
    ts_to_blocks_set = [set(lst) for lst in ts_to_blocks]

    best_solution = None
    best_score = float('inf')

    def coverage_count_from_blocks(current_blocks):
        """Compute coverage count array from a list of block indices."""
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices_np[b_idx]] += 1
        return coverage_count

    def can_remove_block(b_idx, coverage_count, current_blocks_set):
        """Check if block b_idx can be removed (all its t-subsets still covered by others)."""
        subs_in_block = block_t_indices_np[b_idx]
        # For each t-subset in this block, check if another block in current set covers it
        for ts_idx in subs_in_block:
            # Check if any other block covers this t-subset
            others = ts_to_blocks_set[ts_idx]
            # Need at least one other block in current_blocks_set
            found = False
            for other_b in others:
                if other_b != b_idx and other_b in current_blocks_set:
                    found = True
                    break
            if not found:
                return False
        return True

    def delete_blocks(current_blocks, coverage_count):
        """Iteratively remove redundant blocks."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            current_set = set(current_blocks)
            for b_idx in current_blocks:
                subs_in_block = block_t_indices_np[b_idx]
                # Check if all t-subsets have coverage > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                    current_set.discard(b_idx)
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def swap_improve(current_blocks, coverage_count):
        """Try to improve by swapping blocks: remove one, add another, then delete."""
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks, coverage_count

        current_set = set(current_blocks)
        n_current = len(current_blocks)
        improved_any = True

        while improved_any:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved_any = False

            # Try removing each block and see if we can cover the gap with one other block
            for i, b_remove in enumerate(list(current_blocks)):
                if time.time() - start_time > time_limit - 5.0:
                    break

                # What t-subsets become uncovered if we remove b_remove?
                subs_in_block = block_t_indices_np[b_remove]
                # After removal, check which are still covered
                temp_coverage = coverage_count.copy()
                temp_coverage[subs_in_block] -= 1
                uncovered = temp_coverage <= 0
                if not np.any(uncovered):
                    # Can just remove
                    coverage_count = temp_coverage
                    current_blocks = [b for b in current_blocks if b != b_remove]
                    current_set.discard(b_remove)
                    improved_any = True
                    break
                else:
                    # Find uncovered t-subsets
                    uncovered_idx = np.where(uncovered)[0]
                    # For each uncovered t-subset, find candidate blocks that cover it
                    # and check if adding that block covers all uncovered
                    candidates = set()
                    for ts_idx in uncovered_idx:
                        for cand in ts_to_blocks_np[ts_idx]:
                            if cand != b_remove:
                                candidates.add(cand)

                    if len(candidates) > 200:
                        candidates = set(random.sample(list(candidates), 200))

                    best_cand = None
                    best_new_blocks = None
                    best_new_coverage = None

                    for cand in candidates:
                        # Simulate: remove b_remove, add cand
                        cand_subs = block_t_indices_np[cand]
                        # New coverage: temp_coverage + cand's contribution
                        new_cov = temp_coverage.copy()
                        new_cov[cand_subs] += 1
                        if np.all(new_cov > 0):
                            # Valid: try to delete from this new set
                            new_block_list = [b for b in current_blocks if b != b_remove]
                            if cand not in new_block_list:
                                new_block_list.append(cand)
                            # Try deletion
                            new_block_list = delete_blocks(new_block_list, new_cov)
                            if len(new_block_list) < n_current:
                                if best_new_blocks is None or len(new_block_list) < len(best_new_blocks):
                                    best_new_blocks = new_block_list
                                    best_new_coverage = new_cov
                                    best_cand = cand
                                    if len(new_block_list) < n_current - 1:
                                        break  # Found good improvement

                    if best_new_blocks is not None:
                        current_blocks = best_new_blocks
                        coverage_count = best_new_coverage
                        current_set = set(current_blocks)
                        n_current = len(current_blocks)
                        improved_any = True
                        break

        return current_blocks, coverage_count

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        # For vectorized gain computation, we need a different approach
        # Precompute a matrix-like structure: for each t-subset, list of blocks covering it
        # Gain of a block = number of its t-subsets not yet covered

        # Use a different strategy: maintain for each block, how many of its t-subsets are covered
        block_covered_count = np.zeros(n_ks, dtype=np.int32)

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Compute gains: for each block, gain = t_count - block_covered_count[block]
            gains = t_count - block_covered_count

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Among blocks with max gain, pick one that maximizes coverage of rare subsets
            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 100:
                # Sample for speed
                candidates = np.array(random.sample(list(max_gain_blocks), min(100, len(max_gain_blocks))))
            else:
                candidates = max_gain_blocks

            # Score: prefer blocks covering rarer (less frequently covered) t-subsets
            best_score_cand = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices_np[cand]
                newly = ~covered[subs]
                if np.any(newly):
                    # Bonus for covering rarely-covered subsets
                    # Use cover frequency as proxy
                    bonus = np.sum(newly.astype(np.float64)) * 0.5
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus
                if score_cand > best_score_cand:
                    best_score_cand = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            # Update covered and block_covered_count
            subs_to_cover = block_t_indices_np[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            covered[subs_to_cover] = True

            # Update block_covered_count for all blocks that share t-subsets with this block
            # For each t-subset in this block, increment count for all blocks covering it
            for ts_idx in subs_to_cover:
                for other_b in ts_to_blocks_np[ts_idx]:
                    block_covered_count[other_b] += 1

        current_blocks = list(blocks_indices)

        # Build coverage count
        coverage_count = coverage_count_from_blocks(current_blocks)

        # Aggressive deletion
        current_blocks = delete_blocks(current_blocks, coverage_count)

        # Rebuild coverage count after deletion
        coverage_count = coverage_count_from_blocks(current_blocks)

        # Swap-based improvement
        current_blocks, coverage_count = swap_improve(current_blocks, coverage_count)

        # Another deletion pass
        current_blocks = delete_blocks(current_blocks, coverage_count)

        # Final swap pass
        coverage_count = coverage_count_from_blocks(current_blocks)
        current_blocks, coverage_count = swap_improve(current_blocks, coverage_count)

        # Final deletion
        current_blocks = delete_blocks(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
