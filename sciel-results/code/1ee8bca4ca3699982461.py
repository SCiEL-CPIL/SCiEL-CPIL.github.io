"""Combined: NumPy-vectorized greedy (insp1 structure) + scarcity sampling & deletion (current refinement)."""
import itertools
import time
import random
import heapq
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute all k-subsets (blocks) and their t-subset index lists
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)

    # For each block, the list of t-subset indices it covers
    block_t_indices_list = []
    for b in all_k_subsets:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_t_indices_list.append(indices)

    # Convert to numpy array for vectorized operations
    block_t_indices = np.array(block_t_indices_list, dtype=np.int32)

    # For each t-subset, which blocks cover it (for swap candidates)
    t_sub_to_blocks = defaultdict(list)
    for b_idx, indices in enumerate(block_t_indices_list):
        for t_idx in indices:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Point frequency in uncovered t-subsets (for scarcity weighting)
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 40

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)

        covered = np.zeros(n_t, dtype=bool)
        coverage_counts = np.zeros(n_t, dtype=np.int32)
        covered_count = 0
        solution_blocks = []  # list of block indices

        # === Phase 1: Greedy with scarcity-weighted seeding (current program's idea) ===
        while covered_count < n_t and time_ok():
            # Find uncovered t-subsets
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break

            # Vectorized gain computation over all blocks
            # gain[b] = number of uncovered t-subsets covered by block b
            gains = np.zeros(n_blocks, dtype=np.int32)
            for b_idx in range(n_blocks):
                t_idxs = block_t_indices[b_idx]
                gains[b_idx] = (~covered[t_idxs]).sum()

            # If no block covers anything new, break
            if gains.max() == 0:
                break

            # Scarcity-weighted selection among top-gain blocks
            # Compute point scarcity: for each point, count uncovered t-subsets containing it
            point_uncovered_count = np.zeros(v, dtype=np.int32)
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Among blocks with max gain, prefer those with rare points
            max_gain = gains.max()
            top_blocks = np.where(gains == max_gain)[0]

            if len(top_blocks) <= 5:
                chosen = top_blocks[random.randint(0, len(top_blocks) - 1)]
            else:
                # Score each top block by sum of inverse scarcity of its points
                scores = np.zeros(len(top_blocks))
                for i, b_idx in enumerate(top_blocks):
                    b = all_k_subsets[b_idx]
                    s = 0.0
                    for pt in b:
                        s += 1.0 / max(1, point_uncovered_count[pt])
                    scores[i] = s
                # Weighted random selection
                total_s = scores.sum()
                if total_s == 0:
                    chosen = top_blocks[random.randint(0, len(top_blocks) - 1)]
                else:
                    probs = scores / total_s
                    chosen = top_blocks[random.choices(range(len(top_blocks)), p=probs, k=1)[0]]

            # Apply the chosen block
            t_idxs = block_t_indices[chosen]
            newly_covered = t_idxs[~covered[t_idxs]]
            for idx in newly_covered:
                covered[idx] = True
                coverage_counts[idx] += 1
            covered_count += len(newly_covered)
            for idx in t_idxs:
                coverage_counts[idx] += 1
            solution_blocks.append(chosen)

        # === Phase 2: Deletion (current program's priority queue approach) ===
        if time_ok() and covered_count == n_t:
            tsub_block_count = coverage_counts.copy()

            def unique_count(block_idx):
                b = all_k_subsets[block_idx]
                uc = 0
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    if tsub_block_count[idx] == 1:
                        uc += 1
                return uc

            # Build initial priority queue
            pq = []
            for i, b_idx in enumerate(solution_blocks):
                uc = unique_count(b_idx)
                heapq.heappush(pq, (uc, i))

            changed = True
            while changed and time_ok():
                changed = False
                while pq:
                    uc, i = heapq.heappop(pq)
                    if i >= len(solution_blocks):
                        continue
                    b_idx = solution_blocks[i]
                    b = all_k_subsets[b_idx]
                    removable = True
                    for sub in itertools.combinations(b, t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for sub in itertools.combinations(b, t):
                            idx = t_sub_to_idx[sub]
                            tsub_block_count[idx] -= 1
                        solution_blocks.pop(i)
                        changed = True
                        # Rebuild priority queue
                        pq = []
                        for j, bj in enumerate(solution_blocks):
                            uc2 = unique_count(bj)
                            heapq.heappush(pq, (uc2, j))
                        break
                if not pq:
                    break

        # === Phase 3: Swap-based local search (inspiration 1's refinement) ===
        if time_ok() and covered_count == n_t:
            # Update coverage counts to match current solution
            coverage_counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in solution_blocks:
                t_idxs = block_t_indices[b_idx]
                coverage_counts[t_idxs] += 1

            in_solution = np.zeros(n_blocks, dtype=bool)
            for b_idx in solution_blocks:
                in_solution[b_idx] = True

            def try_remove(A_idx):
                """Try to remove block A_idx. Returns True if successful."""
                if A_idx not in solution_blocks:
                    return False
                t_idxs_A = block_t_indices[A_idx]
                if np.any(coverage_counts[t_idxs_A] == 1):
                    return False
                coverage_counts[t_idxs_A] -= 1
                in_solution[A_idx] = False
                solution_blocks.remove(A_idx)
                return True

            def try_swap(A_idx, B_idx):
                """Try to replace block A_idx with B_idx. Returns True if successful."""
                if B_idx in solution_blocks or A_idx not in solution_blocks:
                    return False
                t_idxs_A = block_t_indices[A_idx]
                t_idxs_B = block_t_indices[B_idx]

                essential_mask = (coverage_counts[t_idxs_A] == 1)
                essential_t_idxs = t_idxs_A[essential_mask]

                if len(essential_t_idxs) > 0:
                    b_set = set(t_idxs_B.tolist())
                    if not all(t in b_set for t in essential_t_idxs):
                        return False

                coverage_counts[t_idxs_A] -= 1
                in_solution[A_idx] = False
                solution_blocks.remove(A_idx)

                coverage_counts[t_idxs_B] += 1
                in_solution[B_idx] = True
                solution_blocks.append(B_idx)
                return True

            no_improvement = 0
            while time_ok() and no_improvement < 8:
                improved = False

                # Pass 1: Remove redundant blocks
                for b_idx in solution_blocks[:]:
                    if try_remove(b_idx):
                        improved = True
                        no_improvement = 0
                        break

                if improved:
                    continue

                # Pass 2: Try swaps
                sol_copy = solution_blocks[:]
                random.shuffle(sol_copy)

                for A_idx in sol_copy:
                    if not time_ok():
                        break

                    t_idxs_A = block_t_indices[A_idx]
                    essential_mask = (coverage_counts[t_idxs_A] == 1)
                    essential_t_idxs = t_idxs_A[essential_mask]

                    if len(essential_t_idxs) == 0:
                        continue

                    # Find candidate blocks that cover all essential t-subsets
                    candidates = None
                    for t_idx in essential_t_idxs:
                        blocks_covering = t_sub_to_blocks[t_idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                        if not candidates:
                            break

                    if not candidates:
                        continue

                    candidates = [b for b in candidates if not in_solution[b]]
                    if not candidates:
                        continue

                    random.shuffle(candidates)

                    for B_idx in candidates[:3]:
                        if try_swap(A_idx, B_idx):
                            improved = True
                            no_improvement = 0
                            break

                    if improved:
                        break

                if not improved:
                    no_improvement += 1

        # Update best
        if len(solution_blocks) < best_count:
            best_count = len(solution_blocks)
            best_blocks = list(solution_blocks)

        # Early stop
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(all_k_subsets[b_idx]) for b_idx in best_blocks]
