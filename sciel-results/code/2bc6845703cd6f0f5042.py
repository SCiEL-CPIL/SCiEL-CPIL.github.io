import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (150s limit, leave 38s buffer)

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Build numpy incidence matrix: incidence[i, j] = True if block i covers triple j
    # Also build masks for fast set operations
    masks = [0] * n_k
    incidence = np.zeros((n_k, n_t), dtype=np.uint8)
    block_t_indices_all = [[] for _ in range(n_k)]

    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            incidence[i, idx] = 1
            t_indices.append(idx)
        masks[i] = m
        block_t_indices_all[i] = t_indices

    full_mask = (1 << n_t) - 1

    # Helper: Greedy cover for a specific uncovered mask using numpy
    # Uses integer mask operations to avoid converting to numpy arrays each iteration
    def greedy_cover(uncovered_mask, rng=None):
        if rng is None:
            rng = random.Random()

        selected = []
        current_uncovered = uncovered_mask

        while current_uncovered:
            # Compute gain for all blocks using bit operations
            # We need to count how many bits in (incidence_block & current_uncovered) are set
            # For speed, we'll iterate through blocks and compute gain
            max_gain = 0
            best_indices = []
            
            # Use a faster approach: pre-compute which blocks cover which triples
            # and for each block, count overlap with current_uncovered
            for i in range(n_k):
                if masks[i] & current_uncovered:
                    # Count bits in (masks[i] & current_uncovered)
                    gain = (masks[i] & current_uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_indices = [i]
                    elif gain == max_gain:
                        best_indices.append(i)
            
            if max_gain == 0:
                break

            if len(best_indices) == 1:
                best_idx = best_indices[0]
            else:
                best_idx = best_indices[rng.randint(0, len(best_indices) - 1)]

            selected.append(best_idx)
            current_uncovered &= ~masks[best_idx]

            if time.time() > deadline - 3:
                break

        return selected

    def reduce_cover(blocks):
        if not blocks:
            return []

        cov_counts = [0] * n_t
        current_blocks = list(blocks)
        current_block_t_indices = [block_t_indices_all[b_idx] for b_idx in current_blocks]

        for b_idx in current_blocks:
            for idx in block_t_indices_all[b_idx]:
                cov_counts[idx] += 1

        improved = True
        rng_reduce = random.Random(12345)
        while improved:
            improved = False
            order = list(range(len(current_blocks)))
            rng_reduce.shuffle(order)

            for bi in order:
                if bi >= len(current_blocks):
                    continue

                redundant = True
                for idx in current_block_t_indices[bi]:
                    if cov_counts[idx] == 1:
                        redundant = False
                        break

                if redundant:
                    for idx in current_block_t_indices[bi]:
                        cov_counts[idx] -= 1
                    current_blocks.pop(bi)
                    current_block_t_indices.pop(bi)
                    improved = True
                    break

        return current_blocks

    def is_valid_cover(blocks):
        covered = 0
        for b_idx in blocks:
            covered |= masks[b_idx]
        return covered == full_mask

    # Main ILS Loop
    rng_main = random.Random(42)

    # 1. Initial Greedy with more random restarts, keep best
    best_blocks = None
    best_len = float('inf')

    for restart in range(10):
        if time.time() > deadline - 40:
            break
        rng_init = random.Random(42 + restart * 7)
        init_sel = greedy_cover(full_mask, rng=rng_init)
        reduced = reduce_cover(init_sel)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced

    # 2. ILS perturbation-repair loop
    no_improve_count = 0
    while time.time() < deadline - 15:
        current_blocks = list(best_blocks)

        # Adaptive perturbation: remove blocks with lowest redundancy
        n_remove = rng_main.randint(1, 4)
        if len(current_blocks) <= n_remove:
            continue

        # Score each block by how many triples it uniquely covers (low = good candidate to remove)
        cov_counts = [0] * n_t
        for b_idx in current_blocks:
            for idx in block_t_indices_all[b_idx]:
                cov_counts[idx] += 1

        block_unique = []
        for b_idx in current_blocks:
            u = 0
            for idx in block_t_indices_all[b_idx]:
                if cov_counts[idx] == 1:
                    u += 1
            block_unique.append(u)

        # Prefer removing blocks with fewer unique triples
        # Weighted random selection: lower unique count -> higher probability
        weights = [1.0 / (1.0 + u) for u in block_unique]
        total_w = sum(weights)
        indices_to_remove = set()
        candidates = list(range(len(current_blocks)))
        for _ in range(n_remove):
            if not candidates:
                break
            r = rng_main.random() * total_w
            cum = 0
            chosen = candidates[0]
            for ci in candidates:
                cum += weights[ci]
                if r <= cum:
                    chosen = ci
                    break
            indices_to_remove.add(chosen)
            candidates.remove(chosen)
            total_w -= weights[chosen]

        removed_mask = 0
        new_current = []
        for i, b_idx in enumerate(current_blocks):
            if i in indices_to_remove:
                removed_mask |= masks[b_idx]
            else:
                new_current.append(b_idx)

        covered_by_current = 0
        for b_idx in new_current:
            covered_by_current |= masks[b_idx]

        uncovered = removed_mask & ~covered_by_current

        if uncovered == 0:
            if len(new_current) < best_len:
                best_len = len(new_current)
                best_blocks = new_current
                no_improve_count = 0
            continue

        repair_sel = greedy_cover(uncovered, rng=rng_main)
        candidate_blocks = new_current + repair_sel
        candidate_blocks = reduce_cover(candidate_blocks)

        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_blocks = candidate_blocks
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try a stronger perturbation
            if no_improve_count > 15:
                no_improve_count = 0

    # 3. Final polish: single removals + 1-for-1 swaps
    if time.time() < deadline - 2:
        # Single removals (multiple passes)
        for _pass in range(5):
            removed_any = False
            for i in range(len(best_blocks)):
                if time.time() > deadline - 2:
                    break
                trial = best_blocks[:i] + best_blocks[i+1:]
                covered = 0
                for b_idx in trial:
                    covered |= masks[b_idx]
                if covered == full_mask:
                    best_blocks = trial
                    best_len -= 1
                    removed_any = True
                    break
            if not removed_any:
                break

        # 1-for-1 swaps: replace one block with another to maintain or improve coverage
        if time.time() < deadline - 2:
            for _swap_pass in range(3):
                if time.time() > deadline - 2:
                    break
                found_swap = False
                for i in range(len(best_blocks)):
                    if time.time() > deadline - 2:
                        break
                    # Remove block i, see what's uncovered
                    without_i = best_blocks[:i] + best_blocks[i+1:]
                    covered_without = 0
                    for b_idx in without_i:
                        covered_without |= masks[b_idx]
                    uncovered_by_i = masks[best_blocks[i]] & ~covered_without

                    if uncovered_by_i == 0:
                        # Block i is redundant (shouldn't happen after reduction, but check)
                        best_blocks = without_i
                        best_len -= 1
                        found_swap = True
                        break

                    # Try to find a single block that covers all of uncovered_by_i
                    # and doesn't introduce too much overlap
                    best_swap = -1
                    best_overlap = float('inf')
                    for j in range(n_k):
                        if j == best_blocks[i]:
                            continue
                        if j in best_blocks:
                            continue
                        if (masks[j] & uncovered_by_i) != uncovered_by_i:
                            continue
                        # This block covers all uncovered triples
                        # Count how many of its triples are already covered (waste)
                        already_covered = masks[j] & covered_without
                        waste = already_covered.bit_count()
                        if waste < best_overlap:
                            best_overlap = waste
                            best_swap = j

                    if best_swap != -1:
                        # Check if this swap actually helps (it maintains coverage)
                        # The size stays the same, but we might enable further removals
                        # Only accept if it reduces total redundancy
                        new_blocks = without_i + [best_swap]
                        # Quick check: is there now a redundant block?
                        cov_counts2 = [0] * n_t
                        for b_idx in new_blocks:
                            for idx in block_t_indices_all[b_idx]:
                                cov_counts2[idx] += 1
                        has_redundant = False
                        for b_idx in new_blocks:
                            for idx in block_t_indices_all[b_idx]:
                                if cov_counts2[idx] == 1:
                                    has_redundant = False
                                    break
                            else:
                                has_redundant = True
                                break
                        if has_redundant:
                            reduced = reduce_cover(new_blocks)
                            if len(reduced) < best_len:
                                best_blocks = reduced
                                best_len = len(reduced)
                                found_swap = True
                                break
                if not found_swap:
                    break

    # Final validation and return
    if best_blocks is None or not is_valid_cover(best_blocks):
        # Fallback: full greedy
        rng_fb = random.Random(999)
        best_blocks = greedy_cover(full_mask, rng=rng_fb)
        best_blocks = reduce_cover(best_blocks)

    return [list(all_k_subsets[i]) for i in best_blocks]
