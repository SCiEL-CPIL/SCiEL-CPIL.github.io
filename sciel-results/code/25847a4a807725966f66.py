import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Precompute all t-subsets and their index
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # Precompute all k-subsets and their coverage masks
    # For v=19, k=8: 75582 blocks, each covers C(8,5)=56 t-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    masks = [0] * num_k_subsets
    mask_indices = [[] for _ in range(num_k_subsets)]
    blocks_covering_t = [[] for _ in range(num_t_subsets)]

    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
            blocks_covering_t[idx].append(i)
        masks[i] = mask
        mask_indices[i] = indices

    best_blocks = None
    best_count = float('inf')

    # Density-driven block generation:
    # Instead of pure greedy, we use a scoring function that balances
    # coverage of uncovered t-subsets with diversity to escape local minima.
    # We also add a "block quality" heuristic: prefer blocks that cover
    # t-subsets with fewer alternative covers (critical t-subsets).

    def score_block(block_idx, uncovered_mask, coverage_counts):
        """Score a block based on uncovered coverage and criticality."""
        gain = (masks[block_idx] & uncovered_mask).bit_count()
        if gain == 0:
            return 0
        # Bonus for covering critical t-subsets (few alternatives)
        critical_bonus = 0
        for idx in mask_indices[block_idx]:
            if (uncovered_mask >> idx) & 1:
                # This t-subset is uncovered
                num_covers = len(blocks_covering_t[idx])
                if num_covers <= 3:
                    critical_bonus += (4 - num_covers) * 2
        return gain * 10 + critical_bonus

    # Run multiple random restarts with enhanced local search
    num_restarts = 800

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Enhanced Greedy Construction with density-driven scoring
        uncovered = (1 << num_t_subsets) - 1
        current_blocks_indices = []
        used_blocks = set()
        coverage_counts = [0] * num_t_subsets

        while uncovered and time.time() < deadline:
            # Find an uncovered t-subset index
            if uncovered == 0:
                break
            t_idx = (uncovered & -uncovered).bit_length() - 1

            # Get candidates: all k-blocks containing this t-subset
            candidates = blocks_covering_t[t_idx]

            max_score = 0
            best_candidates = []

            for c in candidates:
                if c in used_blocks:
                    continue
                sc = score_block(c, uncovered, coverage_counts)
                if sc > max_score:
                    max_score = sc
                    best_candidates = [c]
                elif sc == max_score and sc > 0:
                    best_candidates.append(c)

            if max_score == 0:
                break

            # Pick random from best candidates (add diversity)
            chosen_idx = random.choice(best_candidates)
            current_blocks_indices.append(chosen_idx)
            used_blocks.add(chosen_idx)

            # Update coverage counts
            for idx in mask_indices[chosen_idx]:
                coverage_counts[idx] += 1

            uncovered &= ~masks[chosen_idx]

            if uncovered == 0:
                break

        if uncovered != 0:
            continue

        # Local Search with multiple phases
        local_indices = list(current_blocks_indices)

        # Reinitialize coverage counts
        coverage_counts = [0] * num_t_subsets
        for bi in local_indices:
            for idx in mask_indices[bi]:
                coverage_counts[idx] += 1

        improved = True
        max_local_iter = 200
        local_iter = 0

        while improved and time.time() < deadline and local_iter < max_local_iter:
            improved = False
            local_iter += 1

            # Phase 1: Remove redundant blocks
            for pos in range(len(local_indices)):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]
                is_redundant = True
                for idx in mask_indices[bi]:
                    if coverage_counts[idx] < 2:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in mask_indices[bi]:
                        coverage_counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: 1-for-1 Swaps with density-driven scoring
            n_blocks = len(local_indices)
            for pos in range(n_blocks):
                if time.time() > deadline:
                    break
                bi = local_indices[pos]

                # Find t-subsets covered ONLY by bi
                critical_ts = []
                for idx in mask_indices[bi]:
                    if coverage_counts[idx] == 1:
                        critical_ts.append(idx)

                if not critical_ts:
                    # Redundant, remove
                    for idx in mask_indices[bi]:
                        coverage_counts[idx] -= 1
                    local_indices.pop(pos)
                    improved = True
                    break

                # Find a replacement block that covers all critical_ts
                candidates = set(blocks_covering_t[critical_ts[0]])

                for idx in critical_ts[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(blocks_covering_t[idx])

                if candidates:
                    valid_candidates = [c for c in candidates if c not in local_indices]
                    if valid_candidates:
                        # Score candidates by how many additional t-subsets they cover
                        best_sc = -1
                        best_c = None
                        for c in valid_candidates:
                            # Count how many t-subsets in c have coverage_count == 1
                            # (i.e., would become uncovered if we remove bi)
                            additional = 0
                            for idx in mask_indices[c]:
                                if idx not in critical_ts and coverage_counts[idx] >= 1:
                                    additional += 1
                            sc = additional * 10 - len(critical_ts)
                            if sc > best_sc:
                                best_sc = sc
                                best_c = c

                        if best_c is not None:
                            new_bi = best_c
                            # Perform swap
                            for idx in mask_indices[bi]:
                                coverage_counts[idx] -= 1
                            for idx in mask_indices[new_bi]:
                                coverage_counts[idx] += 1
                            local_indices[pos] = new_bi
                            improved = True
                            break

            # Phase 3: 2-for-1 swaps (replace 2 blocks with 1)
            if not improved and len(local_indices) > 5:
                for i in range(len(local_indices)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(local_indices)):
                        if time.time() > deadline:
                            break
                        bi1 = local_indices[i]
                        bi2 = local_indices[j]

                        # Check if bi1 and bi2 together can be replaced by one block
                        # Find t-subsets covered only by bi1 or only by bi2
                        only1 = []
                        only2 = []
                        both = []
                        for idx in mask_indices[bi1]:
                            if coverage_counts[idx] == 1:
                                only1.append(idx)
                            elif coverage_counts[idx] == 2:
                                both.append(idx)
                        for idx in mask_indices[bi2]:
                            if idx in both:
                                continue
                            if coverage_counts[idx] == 1:
                                only2.append(idx)

                        if not only1 and not only2:
                            # Both are redundant
                            for idx in mask_indices[bi1]:
                                coverage_counts[idx] -= 1
                            for idx in mask_indices[bi2]:
                                coverage_counts[idx] -= 1
                            local_indices.pop(j)
                            local_indices.pop(i)
                            improved = True
                            break

                        if only1 and not only2:
                            # bi2 is redundant
                            for idx in mask_indices[bi2]:
                                coverage_counts[idx] -= 1
                            local_indices.pop(j)
                            improved = True
                            break

                        if only2 and not only1:
                            # bi1 is redundant
                            for idx in mask_indices[bi1]:
                                coverage_counts[idx] -= 1
                            local_indices.pop(i)
                            improved = True
                            break

                        # Try to find a single block that covers all critical t-subsets
                        all_critical = only1 + only2
                        if all_critical:
                            candidates = set(blocks_covering_t[all_critical[0]])
                            for idx in all_critical[1:]:
                                if not candidates:
                                    break
                                candidates.intersection_update(blocks_covering_t[idx])

                            if candidates:
                                valid_candidates = [c for c in candidates
                                                    if c not in local_indices]
                                if valid_candidates:
                                    new_bi = random.choice(valid_candidates)
                                    for idx in mask_indices[bi1]:
                                        coverage_counts[idx] -= 1
                                    for idx in mask_indices[bi2]:
                                        coverage_counts[idx] -= 1
                                    for idx in mask_indices[new_bi]:
                                        coverage_counts[idx] += 1
                                    local_indices[i] = new_bi
                                    local_indices.pop(j)
                                    improved = True
                                    break

                    if improved:
                        break

        # Update best
        if len(local_indices) < best_count:
            best_count = len(local_indices)
            best_blocks = [list(k_subsets[i]) for i in local_indices]

    if best_blocks is None:
        # Fallback: simple greedy
        if time.time() < deadline - 5:
            uncovered = (1 << num_t_subsets) - 1
            current_blocks_indices = []
            used_blocks = set()
            while uncovered:
                if time.time() > deadline:
                    break
                t_idx = (uncovered & -uncovered).bit_length() - 1
                candidates = blocks_covering_t[t_idx]
                max_gain = 0
                best_candidates = []
                for c in candidates:
                    if c in used_blocks:
                        continue
                    gain = (masks[c] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [c]
                    elif gain == max_gain and gain > 0:
                        best_candidates.append(c)
                if max_gain == 0:
                    break
                chosen_idx = random.choice(best_candidates)
                current_blocks_indices.append(chosen_idx)
                used_blocks.add(chosen_idx)
                uncovered &= ~masks[chosen_idx]
            if uncovered == 0:
                best_blocks = [list(k_subsets[i]) for i in current_blocks_indices]

    return best_blocks
