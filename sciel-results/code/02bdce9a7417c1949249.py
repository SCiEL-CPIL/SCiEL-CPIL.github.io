import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145

    def elapsed():
        return time.time() - start_time

    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)

    # Precompute block -> t-subset indices
    block_t_indices = []
    for block in k_blocks:
        covered = []
        for sub in itertools.combinations(block, t):
            covered.append(t_index[sub])
        block_t_indices.append(covered)

    # Precompute t-subset -> block indices
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].append(b_idx)

    # Precompute block masks (for fast critical-subset checking)
    # Each block covers C(8,5)=56 t-subsets, so masks fit in Python big ints
    block_masks = []
    for b_idx in range(num_k):
        mask = 0
        for s_idx in block_t_indices[b_idx]:
            mask |= (1 << s_idx)
        block_masks.append(mask)

    # Dual variables: weight for each t-subset based on scarcity
    # Initial: 1 / count of blocks covering it (rare subsets get higher weight)
    t_weights = np.zeros(num_t, dtype=np.float64)
    for i in range(num_t):
        cnt = len(t_to_blocks[i])
        if cnt > 0:
            t_weights[i] = 1.0 / cnt

    # Precompute static gain for each block
    static_gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_indices[i]:
            static_gain[i] = np.sum(t_weights[block_t_indices[i]])

    def greedy_with_noise(noise_scale, seed_val):
        """Run greedy set cover with random noise on gains. Returns selected list."""
        rng = random.Random(seed_val)
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        gain = static_gain.copy()
        if noise_scale > 0:
            noise = np.array([rng.gauss(0, noise_scale) for _ in range(num_k)])
            gain = gain + noise

        while uncovered_count > 0 and elapsed() < deadline - 20:
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(current_gains))

            if gain[best_idx] <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w

        return selected

    def remove_redundant(selected):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not selected:
            return selected
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in selected:
            for s_idx in block_t_indices[b_idx]:
                cover_count[s_idx] += 1

        changed = True
        while changed and elapsed() < deadline - 8:
            changed = False
            new_selected = []
            for b_idx in selected:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cover_count[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cover_count[s_idx] -= 1
            if changed:
                selected = new_selected
        return selected

    def local_search(selected, max_swap_rounds=10):
        """1-for-1 swap local search using bitmask for fast candidate filtering."""
        if not selected:
            return selected
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in selected:
            for s_idx in block_t_indices[b_idx]:
                cover_count[s_idx] += 1

        in_sol = np.zeros(num_k, dtype=np.bool_)
        for b_idx in selected:
            in_sol[b_idx] = True

        def try_drop():
            changed = True
            while changed and elapsed() < deadline - 5:
                changed = False
                new_selected = []
                for b_idx in selected:
                    needed = False
                    for s_idx in block_t_indices[b_idx]:
                        if cover_count[s_idx] == 1:
                            needed = True
                            break
                    if needed:
                        new_selected.append(b_idx)
                    else:
                        changed = True
                        for s_idx in block_t_indices[b_idx]:
                            cover_count[s_idx] -= 1
                        in_sol[b_idx] = False
                if changed:
                    selected[:] = new_selected
                    return True
            return False

        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            if try_drop():
                improved = True
                continue

            # Try swaps: pick a block to remove, find candidates covering its unique subsets
            for i in range(len(selected)):
                if elapsed() > deadline - 5:
                    break
                b_out = selected[i]
                unique_subs = []
                for s_idx in block_t_indices[b_out]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)

                if not unique_subs:
                    continue

                # Build critical mask
                critical_mask = 0
                for s_idx in unique_subs:
                    critical_mask |= (1 << s_idx)

                # Find candidate blocks: must cover all critical subsets
                # Start from the most constrained critical subset (fewest blocks cover it)
                min_s = unique_subs[0]
                min_cnt = len(t_to_blocks[min_s])
                for s_idx in unique_subs[1:]:
                    if len(t_to_blocks[s_idx]) < min_cnt:
                        min_cnt = len(t_to_blocks[s_idx])
                        min_s = s_idx

                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= set(t_to_blocks[s_idx])

                if not candidates:
                    continue
                candidates.discard(b_out)
                candidates = [b for b in candidates if not in_sol[b]]
                if not candidates:
                    continue

                # Score candidates: minimize redundant coverage, maximize new unique coverage
                best_in = None
                best_score = float('inf')
                best_new_unique = -1
                for b_in in candidates:
                    redundant = 0
                    new_unique = 0
                    for s_idx in block_t_indices[b_in]:
                        if cover_count[s_idx] > 1:
                            redundant += 1
                        elif cover_count[s_idx] == 0:
                            new_unique += 1
                    score = redundant
                    if score < best_score or (score == best_score and new_unique > best_new_unique):
                        best_score = score
                        best_new_unique = new_unique
                        best_in = b_in

                if best_in is None:
                    continue

                # Perform swap
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                in_sol[b_out] = False
                for s_idx in block_t_indices[best_in]:
                    cover_count[s_idx] += 1
                in_sol[best_in] = True
                selected[i] = best_in

                if try_drop():
                    improved = True
                    break
                else:
                    # Revert
                    for s_idx in block_t_indices[best_in]:
                        cover_count[s_idx] -= 1
                    in_sol[best_in] = False
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    in_sol[b_out] = True
                    selected[i] = b_out

        return selected

    def randomized_local_search(selected, seed_val, max_rounds=30):
        """Randomized 1-for-1 swap local search for diversity."""
        if not selected:
            return selected
        rng = random.Random(seed_val)
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in selected:
            for s_idx in block_t_indices[b_idx]:
                cover_count[s_idx] += 1

        in_sol = np.zeros(num_k, dtype=np.bool_)
        for b_idx in selected:
            in_sol[b_idx] = True

        best_sol = list(selected)
        best_size = len(selected)
        no_improve = 0

        for rnd in range(max_rounds):
            if elapsed() > deadline - 5:
                break
            if no_improve > 15:
                break

            improved = False

            # Try to drop a redundant block
            order = list(range(len(selected)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(selected):
                    continue
                b = selected[pos]
                redundant = True
                for s_idx in block_t_indices[b]:
                    if cover_count[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for s_idx in block_t_indices[b]:
                        cover_count[s_idx] -= 1
                    in_sol[b] = False
                    selected.pop(pos)
                    improved = True
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_sol = list(selected)
                    break

            if improved:
                no_improve = 0
                continue

            # Try a random swap
            if selected:
                pos = rng.randint(0, len(selected) - 1)
                b_out = selected[pos]
                unique_subs = []
                for s_idx in block_t_indices[b_out]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)

                if not unique_subs:
                    continue

                critical_mask = 0
                for s_idx in unique_subs:
                    critical_mask |= (1 << s_idx)

                # Find candidates covering all critical subsets
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(t_to_blocks[s_idx])
                    else:
                        candidates &= set(t_to_blocks[s_idx])

                if not candidates:
                    continue
                candidates.discard(b_out)
                candidates = [b for b in candidates if not in_sol[b]]
                if not candidates:
                    continue

                b_in = rng.choice(candidates)

                # Perform swap
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                in_sol[b_out] = False
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                in_sol[b_in] = True
                selected[pos] = b_in
                improved = True
                no_improve = 0
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_sol = list(selected)
            else:
                break

        # Final drop pass
        if selected:
            cover_count2 = np.zeros(num_t, dtype=np.int32)
            for b_idx in selected:
                for s_idx in block_t_indices[b_idx]:
                    cover_count2[s_idx] += 1
            changed = True
            while changed:
                changed = False
                new_sel = []
                for b_idx in selected:
                    needed = False
                    for s_idx in block_t_indices[b_idx]:
                        if cover_count2[s_idx] == 1:
                            needed = True
                            break
                    if needed:
                        new_sel.append(b_idx)
                    else:
                        changed = True
                        for s_idx in block_t_indices[b_idx]:
                            cover_count2[s_idx] -= 1
                if changed:
                    selected = new_sel
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_sol = list(selected)

        return best_sol

    # Main search
    best_selected = None
    best_size = float('inf')

    # Phase 1: Deterministic greedy + local search
    sel = greedy_with_noise(0.0, 42)
    sel = remove_redundant(sel)
    sel = local_search(sel)
    best_selected = sel
    best_size = len(sel)

    # Phase 2: Randomized restarts with varying noise
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for noise in noise_levels:
        if elapsed() > deadline - 30:
            break
        for seed in seeds:
            if elapsed() > deadline - 25:
                break
            sel = greedy_with_noise(noise, seed)
            if len(sel) >= best_size:
                continue
            sel = remove_redundant(sel)
            if len(sel) >= best_size:
                continue
            sel = local_search(sel)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel

    # Phase 3: Randomized local search on best solution for further refinement
    if elapsed() < deadline - 15:
        for seed in [100, 101, 102, 103, 104]:
            if elapsed() > deadline - 10:
                break
            sel = randomized_local_search(list(best_selected), seed, max_rounds=40)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = sel

    # Phase 4: Final high-precision polish
    if elapsed() < deadline - 5:
        best_selected = local_search(best_selected, max_swap_rounds=20)
        best_size = len(best_selected)

    # Final drop pass
    if elapsed() < deadline - 3:
        best_selected = remove_redundant(best_selected)
        best_size = len(best_selected)

    result = [list(k_blocks[i]) for i in best_selected]
    return result
