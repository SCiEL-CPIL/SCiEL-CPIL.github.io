import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s limit

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Build bitmask representation for each block
    # block_mask[bi] = Python int bitmask of which t-subsets are covered
    block_mask = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for s in itertools.combinations(b, t):
            mask |= (1 << t_idx[s])
        block_mask[bi] = mask

    # Build t-subset -> list of blocks covering it
    tsub_to_blocks = [[] for _ in range(N)]
    for bi, b in enumerate(all_blocks):
        for s in itertools.combinations(b, t):
            tsub_to_blocks[t_idx[s]].append(bi)

    # Precompute block marginal coverage sizes (for tie-breaking)
    block_cover_sizes = [bin(m).count('1') for m in block_mask]

    def greedy_bitmask(seed=42):
        """
        Greedy set cover using bitmask representation.
        Always picks the block with max marginal coverage for the hardest uncovered t-subset.
        """
        random.seed(seed)
        uncovered = (1 << N) - 1  # All t-subsets uncovered
        selected = []
        used_blocks = set()

        while uncovered:
            if time.time() > deadline - 25:
                break

            # Find the uncovered t-subset with fewest available covering blocks
            # Sample for speed
            unc_bits = [i for i in range(N) if (uncovered >> i) & 1]
            if not unc_bits:
                break

            sample_size = min(20, len(unc_bits))
            if sample_size < len(unc_bits):
                sample = random.sample(unc_bits, sample_size)
            else:
                sample = unc_bits

            best_target = -1
            min_avail = float('inf')
            for ti in sample:
                avail = 0
                for bi in tsub_to_blocks[ti]:
                    if bi not in used_blocks:
                        avail += 1
                        if avail >= min_avail:
                            break
                if avail < min_avail:
                    min_avail = avail
                    best_target = ti
                    if avail <= 1:
                        break

            if best_target == -1:
                best_target = random.choice(unc_bits)

            # Among blocks covering best_target, pick max marginal gain
            candidates = tsub_to_blocks[best_target]
            best_bi = -1
            best_gain = -1

            cand_list = [bi for bi in candidates if bi not in used_blocks]
            if not cand_list:
                # Fallback: any block
                for bi in candidates:
                    if bi not in used_blocks:
                        best_bi = bi
                        break
                if best_bi == -1:
                    break
            else:
                # Shuffle for diversity
                random.shuffle(cand_list)
                for bi in cand_list:
                    gain = bin(uncovered & block_mask[bi]).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                        if gain > 100:  # Early exit
                            break

            if best_bi == -1:
                break

            selected.append(best_bi)
            used_blocks.add(best_bi)
            uncovered &= ~block_mask[best_bi]

        # Repair any remaining uncovered
        if uncovered:
            for ti in range(N):
                if (uncovered >> ti) & 1:
                    for bi in tsub_to_blocks[ti]:
                        if bi not in used_blocks:
                            selected.append(bi)
                            used_blocks.add(bi)
                            uncovered &= ~block_mask[bi]
                            break

        return selected

    def local_search_101(blocks, max_time=30.0, seed=0):
        """
        1-0-1 local search: try to remove a block, or swap one block for another.
        Uses bitmask for fast coverage checking.
        """
        random.seed(seed)
        start_time = time.time()
        current = list(blocks)
        current_size = len(current)
        best = list(current)
        best_size = current_size

        # Build coverage bitmask: which t-subsets are covered
        cover_mask = 0
        for bi in current:
            cover_mask |= block_mask[bi]

        # Build coverage count using array
        cover_count = np.zeros(N, dtype=np.int16)
        for bi in current:
            # Extract bits
            mask = block_mask[bi]
            for ti in range(N):
                if (mask >> ti) & 1:
                    cover_count[ti] += 1

        current_set = set(current)

        # Precompute block masks as arrays for faster count updates
        block_mask_arrs = []
        for bi in range(num_blocks):
            m = block_mask[bi]
            arr = np.zeros(N, dtype=np.int8)
            for ti in range(N):
                if (m >> ti) & 1:
                    arr[ti] = 1
            block_mask_arrs.append(arr)

        for iteration in range(200000):
            if time.time() > start_time + max_time:
                break

            move_type = random.random()

            if move_type < 0.5:
                # Try removal
                if not current:
                    continue
                idx = random.randint(0, len(current) - 1)
                bi = current[idx]
                arr = block_mask_arrs[bi]

                if np.any(cover_count[arr == 1] == 1):
                    continue

                # Safe to remove
                cover_count[arr == 1] -= 1
                current.pop(idx)
                current_set.remove(bi)
                current_size -= 1

                if current_size < best_size:
                    best_size = current_size
                    best = list(current)

            elif move_type < 0.9:
                # Try swap: remove bi_old, add bi_new
                if not current:
                    continue
                idx = random.randint(0, len(current) - 1)
                bi_old = current[idx]
                arr_old = block_mask_arrs[bi_old]

                # Find t-subsets that would become uncovered
                at_risk = np.where((arr_old == 1) & (cover_count == 1))[0]

                if len(at_risk) == 0:
                    # Just remove
                    cover_count[arr_old == 1] -= 1
                    current.pop(idx)
                    current_set.remove(bi_old)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best = list(current)
                    continue

                # Find candidate blocks covering all at_risk
                candidates = set(tsub_to_blocks[at_risk[0]])
                for ti in at_risk[1:min(len(at_risk), 3)]:
                    candidates &= set(tsub_to_blocks[ti])
                candidates -= current_set

                if not candidates:
                    continue

                cand_list = list(candidates)
                if len(cand_list) > 50:
                    cand_list = random.sample(cand_list, 50)

                best_cand = -1
                best_score = -1
                for cand in cand_list:
                    arr_new = block_mask_arrs[cand]
                    # Score: marginal coverage of new block minus loss from old
                    new_cov = np.sum(cover_count[arr_new == 1] == 0)
                    old_waste = np.sum(cover_count[arr_old == 1] > 1)
                    score = new_cov + old_waste
                    if score > best_score:
                        best_score = score
                        best_cand = cand

                if best_cand == -1:
                    continue

                # Check validity after swap
                new_cover_count = cover_count.copy()
                new_cover_count[arr_old == 1] -= 1
                arr_new = block_mask_arrs[best_cand]
                new_cover_count[arr_new == 1] += 1

                if np.any(new_cover_count < 0):
                    continue

                # Accept if improves waste
                old_waste_total = np.sum(np.maximum(cover_count - 1, 0))
                new_waste_total = np.sum(np.maximum(new_cover_count - 1, 0))
                delta = new_waste_total - old_waste_total

                T = max(0.5, 5.0 * (1 - iteration / 200000))
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.1)):
                    cover_count[arr_old == 1] -= 1
                    cover_count[arr_new == 1] += 1
                    current[idx] = best_cand
                    current_set.remove(bi_old)
                    current_set.add(best_cand)
                    # Size stays same
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)

            else:
                # Try add (to escape local optima)
                if len(current) > 400:
                    continue
                low_cov = np.where(cover_count <= 1)[0]
                if len(low_cov) == 0:
                    continue
                ti = random.choice(low_cov)
                candidates = [c for c in tsub_to_blocks[ti] if c not in current_set]
                if not candidates:
                    continue
                bi_new = random.choice(candidates)
                arr_new = block_mask_arrs[bi_new]

                new_cover_count = cover_count.copy()
                new_cover_count[arr_new == 1] += 1

                old_waste = np.sum(np.maximum(cover_count - 1, 0))
                new_waste = np.sum(np.maximum(new_cover_count - 1, 0))
                delta = 1 + (new_waste - old_waste)

                T = max(0.5, 5.0 * (1 - iteration / 200000))
                if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.1)):
                    cover_count[arr_new == 1] += 1
                    current.append(bi_new)
                    current_set.add(bi_new)
                    current_size += 1

        return best, best_size

    def final_cleanup(blocks, max_time=5.0):
        """Aggressive removal pass."""
        start_time = time.time()
        current = list(blocks)
        current_size = len(current)
        best = list(current)
        best_size = current_size

        cover_count = np.zeros(N, dtype=np.int16)
        for bi in current:
            mask = block_mask[bi]
            for ti in range(N):
                if (mask >> ti) & 1:
                    cover_count[ti] += 1

        block_mask_arrs = []
        for bi in range(num_blocks):
            m = block_mask[bi]
            arr = np.zeros(N, dtype=np.int8)
            for ti in range(N):
                if (m >> ti) & 1:
                    arr[ti] = 1
            block_mask_arrs.append(arr)

        # Multiple passes
        for _pass in range(10):
            if time.time() > start_time + max_time:
                break
            progress = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > start_time + max_time:
                    break
                bi = current[idx]
                arr = block_mask_arrs[bi]
                if np.any(cover_count[arr == 1] == 1):
                    continue
                cover_count[arr == 1] -= 1
                current.pop(idx)
                current_size -= 1
                progress = True
                if current_size < best_size:
                    best_size = current_size
                    best = list(current)
            if not progress:
                break

        return best, best_size

    # Main algorithm
    best_solution = None
    best_size = float('inf')

    # Phase 1: Multiple greedy starts with bitmask
    seeds = [42, 137, 251, 389, 523, 671, 811, 967, 1049, 1181]
    for seed in seeds:
        if time.time() > deadline - 40:
            break

        sol = greedy_bitmask(seed=seed)
        if not sol:
            continue

        cleaned, cleaned_size = final_cleanup(sol, max_time=3.0)
        if cleaned_size < best_size:
            best_size = cleaned_size
            best_solution = cleaned

    if best_solution is None:
        sol = greedy_bitmask(seed=42)
        if sol:
            best_solution = sol
            best_size = len(sol)
        else:
            return [list(b) for b in all_blocks[:100]]

    # Phase 2: 1-0-1 local search on best solution
    remaining_time = deadline - 3 - time.time()
    if remaining_time > 10:
        sa_sol, sa_size = local_search_101(best_solution, max_time=min(remaining_time, 50.0), seed=123)
        if sa_size < best_size:
            best_size = sa_size
            best_solution = sa_sol

    # Phase 3: Run local search on other greedy solutions if time permits
    remaining_time = deadline - 3 - time.time()
    if remaining_time > 15:
        for seed in [777, 888, 999]:
            if time.time() > deadline - 5:
                break
            sol = greedy_bitmask(seed=seed)
            if sol and len(sol) < best_size + 15:
                cleaned, _ = final_cleanup(sol, max_time=2.0)
                sa_sol, sa_size = local_search_101(cleaned, max_time=min(remaining_time / 3, 20.0), seed=seed)
                if sa_size < best_size:
                    best_size = sa_size
                    best_solution = sa_sol

    # Final cleanup
    if time.time() < deadline - 2:
        final, final_size = final_cleanup(best_solution, max_time=3.0)
        if final_size < best_size:
            best_size = final_size
            best_solution = final

    # Convert to point lists
    final_blocks = [list(all_blocks[bi]) for bi in best_solution]

    # Verify validity
    covered_mask = 0
    for bi in best_solution:
        covered_mask |= block_mask[bi]

    if covered_mask != (1 << N) - 1:
        # Repair
        uncovered_mask = ((1 << N) - 1) & ~covered_mask
        used_set = set(best_solution)
        for ti in range(N):
            if (uncovered_mask >> ti) & 1:
                for bi in tsub_to_blocks[ti]:
                    if bi not in used_set:
                        best_solution.append(bi)
                        used_set.add(bi)
                        covered_mask |= block_mask[bi]
                        break
        final_blocks = [list(all_blocks[bi]) for bi in best_solution]

    return final_blocks
