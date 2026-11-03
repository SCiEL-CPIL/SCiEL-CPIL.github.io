import itertools
import time
import random
import math
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all possible blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_to_covered[i] = list of t-subset indices covered by block i
    block_to_covered = []
    for block in all_blocks:
        covered = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_to_covered.append(covered)

    # Reverse index
    t_subset_to_blocks = [set() for _ in range(n_t)]
    for b_idx, covered in enumerate(block_to_covered):
        for ts_idx in covered:
            t_subset_to_blocks[ts_idx].add(b_idx)

    def is_valid_solution(block_indices):
        if not block_indices:
            return n_t == 0
        covered = set()
        for idx in block_indices:
            covered.update(block_to_covered[idx])
        return len(covered) == n_t

    # ---- Column generation LP relaxation ----
    # x[b] for b in active set; constraints: sum_{b covers j} x[b] >= 1 for each j
    def solve_column_generation(max_rounds=200):
        # Start with a small set of blocks from a quick greedy
        active_blocks = set()
        uncovered = set(range(n_t))
        # Quick greedy to seed
        for _ in range(min(50, n_t)):
            if not uncovered:
                break
            j = random.choice(list(uncovered))
            best_b, best_gain = None, -1
            for b in t_subset_to_blocks[j]:
                gain = len(block_to_covered[b] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b is not None:
                active_blocks.add(best_b)
                for ts in block_to_covered[best_b]:
                    uncovered.discard(ts)

        if not active_blocks:
            # Fallback: add blocks covering each t-subset
            for j in range(n_t):
                if t_subset_to_blocks[j]:
                    active_blocks.add(next(iter(t_subset_to_blocks[j])))

        active_blocks = list(active_blocks)
        active_set = set(active_blocks)

        # LP solve via simple subgradient / projected method
        # Variables x[b] >= 0, minimize sum x[b]
        # Constraints: for each j, sum_{b in active, j in block_to_covered[b]} x[b] >= 1
        n_active = len(active_blocks)
        if n_active == 0:
            return []

        # Build constraint matrix A (n_t x n_active) sparsely
        # A[j, i] = 1 if active_blocks[i] covers j
        # Use sparse representation: for each j, list of active indices covering it
        active_idx_to_pos = {b: i for i, b in enumerate(active_blocks)}
        A_rows = [set() for _ in range(n_t)]
        for i, b in enumerate(active_blocks):
            for j in block_to_covered[b]:
                A_rows[j].add(i)

        # Initialize x uniformly
        x = np.ones(n_active, dtype=np.float64)
        # Dual variables y[j] >= 0 for each constraint
        y = np.ones(n_t, dtype=np.float64) * 0.01

        step_size = 0.1
        for round_num in range(max_rounds):
            if time.time() > deadline - 60.0:
                break

            # Compute constraint violations: v_j = 1 - sum_{i: j in block_i} x[i]
            # For each j, violation = 1 - sum(x[i] for i in A_rows[j])
            violations = np.zeros(n_t, dtype=np.float64)
            for j in range(n_t):
                s = 0.0
                for i in A_rows[j]:
                    s += x[i]
                violations[j] = 1.0 - s

            # Update duals: y[j] = max(0, y[j] + step * violations[j])
            for j in range(n_t):
                if violations[j] > 0:
                    y[j] = max(0.0, y[j] + step_size * violations[j])
                else:
                    y[j] = max(0.0, y[j] * 0.95)

            # Pricing: find block b (in all_blocks, not necessarily active) that minimizes
            # reduced cost: 1 - sum_{j in block_to_covered[b]} y[j]
            # We want blocks with sum_{j in covered} y[j] > 1 (negative reduced cost)
            best_b = None
            best_rc = 1.0  # reduced cost = 1 - sum y
            # Sample a subset of blocks for speed
            sample_size = min(20000, n_blocks)
            if n_blocks <= sample_size:
                candidates = range(n_blocks)
            else:
                candidates = random.sample(range(n_blocks), sample_size)
            for b in candidates:
                s = 0.0
                for j in block_to_covered[b]:
                    s += y[j]
                rc = 1.0 - s
                if rc < best_rc:
                    best_rc = rc
                    best_b = b

            if best_b is not None and best_rc < -1e-4:
                # Add new block to active set
                if best_b not in active_set:
                    active_blocks.append(best_b)
                    active_set.add(best_b)
                    n_active += 1
                    # Extend x
                    x = np.append(x, [0.0])
                    # Update A_rows for new block
                    new_pos = n_active - 1
                    for j in block_to_covered[best_b]:
                        A_rows[j].add(new_pos)

            # Update primal x via projected subgradient
            # Gradient of objective w.r.t. x[i] is 1 - sum_{j: i in A_rows[j]} y[j]
            # Actually for min sum x s.t. Ax >= 1, x >= 0:
            # We use a simple approach: project x onto feasible region
            # For each i, x[i] should be at least max over j in block_i of (1 - sum_{i' != i in A_rows[j]} x[i'])
            # This is expensive; instead use multiplicative weights / dual ascent

            # Dual ascent: increase y where constraints violated, decrease where slack
            # Then set x[i] = max(0, 1 - sum_{j not in block_i} ... ) -- too complex
            # Instead: simple heuristic - set x[i] proportional to how much block i is needed
            # For each i, compute "demand": number of constraints where i is the only active cover
            # Simplify: just keep x as is and rely on rounding

            step_size *= 0.99

        # Round the LP solution: use greedy with LP-guided priorities
        # Priority of block b: sum_{j in block_to_covered[b]} y[j] (how valuable it is)
        # Select blocks in order of priority / cost, ensuring coverage
        uncovered = set(range(n_t))
        selected = []
        block_priority = {}
        for i, b in enumerate(active_blocks):
            s = 0.0
            for j in block_to_covered[b]:
                s += y[j]
            block_priority[b] = s

        while uncovered:
            if time.time() > deadline - 45.0:
                break
            # Find block in active set (or all blocks) that covers most uncovered
            best_b, best_gain = None, -1
            # Try active blocks first
            for b in active_blocks:
                gain = len(block_to_covered[b] & uncovered)
                # Weight by priority
                score = gain * (1.0 + block_priority.get(b, 0.0))
                if score > best_gain:
                    best_gain = score
                    best_b = b
            if best_b is None or best_gain <= 0:
                # Fallback: search all blocks
                for b in range(n_blocks):
                    gain = len(block_to_covered[b] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            if best_b is None:
                break
            selected.append(best_b)
            for j in block_to_covered[best_b]:
                uncovered.discard(j)

        return selected

    # ---- Greedy set cover (baseline) ----
    def greedy_cover(seed=0):
        rng = random.Random(seed)
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            if time.time() > deadline - 30.0:
                break
            # Pick random uncovered t-subset
            start_idx = rng.choice(list(uncovered))
            candidate_blocks = t_subset_to_blocks[start_idx]
            best_block = None
            best_gain = -1
            for b_idx in candidate_blocks:
                gain = len(block_to_covered[b_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block is None:
                break
            blocks.append(best_block)
            for ts_idx in block_to_covered[best_block]:
                uncovered.discard(ts_idx)
        return blocks

    # ---- Local search: removal and swap moves ----
    def local_search(block_indices, seed=0, max_steps=5000):
        rng = random.Random(seed)
        current_indices = list(block_indices)
        best_indices = list(current_indices)
        best_len = len(current_indices)

        # Coverage counts
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in current_indices:
            for ts_idx in block_to_covered[idx]:
                counts[ts_idx] += 1

        T = 1.0
        T_min = 0.01
        cooling_rate = 0.97

        step = 0
        while time.time() < deadline - 15.0 and step < max_steps:
            step += 1
            move_type = rng.random()

            if move_type < 0.6 and len(current_indices) > 1:
                # Try to remove a block
                # Find blocks with no critical coverage (all their t-subsets have count >= 2)
                removable = []
                for i, idx in enumerate(current_indices):
                    has_critical = False
                    for ts_idx in block_to_covered[idx]:
                        if counts[ts_idx] == 1:
                            has_critical = True
                            break
                    if not has_critical:
                        removable.append(i)

                if removable:
                    i = rng.choice(removable)
                    idx_to_remove = current_indices[i]
                    # Verify
                    valid = True
                    for ts_idx in block_to_covered[idx_to_remove]:
                        if counts[ts_idx] == 1:
                            valid = False
                            break
                    if valid:
                        current_indices.pop(i)
                        for ts_idx in block_to_covered[idx_to_remove]:
                            counts[ts_idx] -= 1
                        best_indices = list(current_indices)
                        best_len = len(current_indices)
                else:
                    # Try random removals
                    for _ in range(3):
                        i = rng.randint(0, len(current_indices) - 1)
                        idx_to_remove = current_indices[i]
                        valid = True
                        for ts_idx in block_to_covered[idx_to_remove]:
                            if counts[ts_idx] == 1:
                                valid = False
                                break
                        if valid:
                            current_indices.pop(i)
                            for ts_idx in block_to_covered[idx_to_remove]:
                                counts[ts_idx] -= 1
                            best_indices = list(current_indices)
                            best_len = len(current_indices)
                            break

            else:
                # Swap: remove one, add another
                if len(current_indices) == 0:
                    continue
                i = rng.randint(0, len(current_indices) - 1)
                idx_to_remove = current_indices[i]

                # Find candidates that cover low-count t-subsets
                # Sample t-subsets with count <= 2
                low_count = [j for j in range(n_t) if counts[j] <= 2]
                if not low_count:
                    low_count = list(range(n_t))
                sample_size = min(10, len(low_count))
                sampled = rng.sample(low_count, sample_size)

                candidates = set()
                for j in sampled:
                    candidates.update(t_subset_to_blocks[j])
                candidates.discard(idx_to_remove)

                if not candidates:
                    continue

                # Try a few candidates
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                improved = False
                for idx_to_add in cand_list[:5]:
                    if idx_to_add in current_indices:
                        continue
                    # Check validity
                    valid = True
                    for ts_idx in block_to_covered[idx_to_remove]:
                        if counts[ts_idx] == 1 and ts_idx not in block_to_covered[idx_to_add]:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        current_indices[i] = idx_to_add
                        for ts_idx in block_to_covered[idx_to_remove]:
                            counts[ts_idx] -= 1
                        for ts_idx in block_to_covered[idx_to_add]:
                            counts[ts_idx] += 1
                        improved = True
                        break

                if not improved:
                    # Try to find a swap that enables future removal
                    # Just accept a random valid swap for exploration
                    for idx_to_add in cand_list[:3]:
                        if idx_to_add in current_indices:
                            continue
                        # Check if swap is valid (all t-subsets still covered)
                        valid = True
                        for ts_idx in block_to_covered[idx_to_remove]:
                            if counts[ts_idx] == 1 and ts_idx not in block_to_covered[idx_to_add]:
                                valid = False
                                break
                        if valid:
                            current_indices[i] = idx_to_add
                            for ts_idx in block_to_covered[idx_to_remove]:
                                counts[ts_idx] -= 1
                            for ts_idx in block_to_covered[idx_to_add]:
                                counts[ts_idx] += 1
                            break

            T = max(T_min, T * cooling_rate)
            if step % 500 == 0 and T == T_min:
                T = 0.5

        return best_indices, best_len

    # ---- Main: combine column generation + greedy + local search ----
    best_indices = None
    best_len = float('inf')

    # 1. Column generation approach
    if time.time() < deadline - 80.0:
        cg_indices = solve_column_generation(100)
        if cg_indices and is_valid_solution(cg_indices):
            best_indices = cg_indices
            best_len = len(cg_indices)

    # 2. Greedy baseline
    greedy_indices = greedy_cover(0)
    if is_valid_solution(greedy_indices):
        if best_indices is None or len(greedy_indices) < best_len:
            best_indices = greedy_indices
            best_len = len(greedy_indices)

    # 3. Local search on best solution
    if best_indices is not None and time.time() < deadline - 20.0:
        for seed in range(8):
            if time.time() > deadline - 15.0:
                break
            ls_indices, ls_len = local_search(best_indices, seed=seed, max_steps=3000)
            if ls_len < best_len:
                best_len = ls_len
                best_indices = ls_indices

    # 4. Additional local search rounds
    if best_indices is not None and time.time() < deadline - 10.0:
        ls_indices, ls_len = local_search(best_indices, seed=999, max_steps=5000)
        if ls_len < best_len:
            best_len = ls_len
            best_indices = ls_indices

    # Final validation and conversion
    if best_indices is None or not is_valid_solution(best_indices):
        best_indices = greedy_cover(0)

    result = [list(all_blocks[i]) for i in best_indices]

    # Ensure distinctness
    seen = set()
    unique_result = []
    for b in result:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_result.append(b)
    result = unique_result

    # Final check
    if not is_valid_solution([all_blocks.index(tuple(b)) for b in result]):
        result = [list(all_blocks[i]) for i in greedy_cover(0)]

    return result
