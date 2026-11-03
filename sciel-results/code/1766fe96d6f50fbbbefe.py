import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s margin for safety

    # 1. Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    t_in_k = list(itertools.combinations(range(k), t))
    n_t_in_k = len(t_in_k)

    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)

    # Fill block_t_indices
    local_t_combos = t_in_k
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(local_t_combos):
            block_t_indices[i, j] = t_sub_to_idx[tuple(block[p] for p in lc)]

    # 4. Build reverse mapping: for each t-subset, which blocks cover it
    # This is expensive but useful for local search
    # sub_to_blocks[s] = list of block indices covering subset s
    sub_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            sub_to_blocks[block_t_indices[i, j]].append(i)

    if time.time() > deadline - 10.0:
        return []

    # 5. Greedy Set Cover with Multiple Restarts + Local Search
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)

    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1

        # Reset state
        uncovered = np.ones(n_t, dtype=bool)
        selected_blocks = []
        selected_set = set()

        # Greedy loop
        while True:
            if time.time() > deadline - 20.0:
                break

            # Calculate gains: number of uncovered t-subsets covered by each block
            gains = uncovered[block_t_indices].sum(axis=1)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]

            # Pick one randomly to encourage diversity
            best_idx = int(rng.choice(candidates))

            selected_blocks.append(best_idx)
            selected_set.add(best_idx)

            # Update uncovered
            uncovered[block_t_indices[best_idx]] = False

        if time.time() > deadline - 15.0:
            break

        if not selected_blocks:
            continue

        # 6. Local Search Optimization
        # Build coverage counts
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in selected_blocks:
            coverage_count[block_t_indices[bi]] += 1

        # First pass: remove redundant blocks
        remaining = list(selected_blocks)
        changed = True
        while changed:
            if time.time() > deadline - 10.0:
                break
            changed = False
            for i in range(len(remaining)):
                bi = remaining[i]
                ts = block_t_indices[bi]
                if np.all(coverage_count[ts] > 1):
                    coverage_count[ts] -= 1
                    remaining.pop(i)
                    selected_set.discard(bi)
                    changed = True
                    break

        # Second pass: try to swap blocks to reduce count further
        # For each block, try to remove it and find a replacement that covers its critical subsets
        improved = True
        while improved and time.time() < deadline - 8.0:
            improved = False
            for i in range(len(remaining)):
                if time.time() > deadline - 8.0:
                    break

                bi = remaining[i]
                covers_b = block_t_indices[bi]

                # Check if redundant
                if np.all(coverage_count[covers_b] > 1):
                    coverage_count[covers_b] -= 1
                    remaining.pop(i)
                    selected_set.discard(bi)
                    improved = True
                    break

                # Find critical subsets (covered only by this block)
                critical_mask = coverage_count[covers_b] == 1
                if not np.any(critical_mask):
                    continue

                critical_subsets = covers_b[critical_mask]
                n_critical = len(critical_subsets)

                if n_critical == 0:
                    continue

                # Find candidate blocks that cover ALL critical subsets
                # Intersection of sub_to_blocks[s] for s in critical_subsets
                if n_critical == 1:
                    candidates = sub_to_blocks[critical_subsets[0]]
                elif n_critical == 2:
                    c1 = set(sub_to_blocks[critical_subsets[0]])
                    c2 = set(sub_to_blocks[critical_subsets[1]])
                    candidates = list(c1 & c2)
                elif n_critical == 3:
                    c1 = set(sub_to_blocks[critical_subsets[0]])
                    c2 = set(sub_to_blocks[critical_subsets[1]])
                    c3 = set(sub_to_blocks[critical_subsets[2]])
                    candidates = list(c1 & c2 & c3)
                else:
                    # For more critical subsets, use intersection of all
                    if n_critical > 5:
                        # Too many critical subsets, skip this swap attempt
                        continue
                    candidates = set(sub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidates &= set(sub_to_blocks[s])
                        if not candidates:
                            break
                    candidates = list(candidates)

                if not candidates:
                    continue

                # Filter out already selected blocks
                candidates = [c for c in candidates if c not in selected_set]
                if not candidates:
                    continue

                # Pick the best candidate: one that maximizes "waste reduction"
                # i.e., covers the most already-covered subsets (so we gain less new coverage)
                # Actually, we want to minimize the net change in coverage
                # A good heuristic: pick the candidate that covers the most subsets already covered
                best_cand = -1
                best_score = -1
                for c in candidates[:20]:  # Limit candidates checked
                    covers_c = block_t_indices[c]
                    # Count how many of covers_c are already covered (count > 0)
                    already_covered = np.sum(coverage_count[covers_c] > 0)
                    # We want to maximize already_covered to minimize waste
                    if already_covered > best_score:
                        best_score = already_covered
                        best_cand = c

                if best_cand != -1:
                    # Perform the swap
                    # Remove bi
                    coverage_count[covers_b] -= 1
                    remaining.pop(i)
                    selected_set.discard(bi)

                    # Add best_cand
                    covers_new = block_t_indices[best_cand]
                    coverage_count[covers_new] += 1
                    remaining.append(best_cand)
                    selected_set.add(best_cand)
                    improved = True
                    break

        if time.time() > deadline - 8.0:
            break

        # Final check: verify the solution is valid
        if remaining:
            # Verify coverage
            test_coverage = np.zeros(n_t, dtype=np.int32)
            for bi in remaining:
                test_coverage[block_t_indices[bi]] += 1
            if np.all(test_coverage > 0):
                if len(remaining) < best_size:
                    best_size = len(remaining)
                    best_solution = [k_subsets[i] for i in remaining]

    if best_solution is None:
        return []

    return [list(b) for b in best_solution]
