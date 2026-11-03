import numpy as np
import itertools
import time
import random
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    c_k_t = comb(k, t)

    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
        block_triples[i] = np.sort(block_triples[i])

    triple_counts = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_counts[block_triples[i]] += 1

    triple_weights = 1.0 / (triple_counts + 1).astype(np.float64)

    triple_to_blocks = [np.empty(0, dtype=np.int32) for _ in range(n_triples)]
    for i in range(n_triples):
        triple_to_blocks[i] = np.empty(triple_counts[i], dtype=np.int32)
    cursors = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        for triple_idx in block_triples[i]:
            triple_to_blocks[triple_idx][cursors[triple_idx]] = i
            cursors[triple_idx] += 1

    block_sets = [set(block) for block in all_blocks]
    block_tuples = [tuple(block) for block in all_blocks]
    tuple_to_idx = {bt: i for i, bt in enumerate(block_tuples)}

    # Precompute block membership as a boolean matrix for faster overlap checks
    # block_contains_triple[i, j] = 1 if block i contains triple j
    block_contains = np.zeros((n_blocks, n_triples), dtype=np.int8)
    for i in range(n_blocks):
        block_contains[i, block_triples[i]] = 1

    def dynamic_greedy_cover(random_tiebreak=False, weight_power=1.0, rng=None,
                              scarcity_mode='dynamic'):
        """
        Dynamic scarcity-weighted greedy: update triple weights as triples get
        covered, prioritizing triples that become increasingly scarce (fewer
        remaining blocks can cover them).
        """
        if rng is None:
            rng = random.Random(42)
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []

        # Dynamic weights: track how many blocks can still cover each uncovered triple
        # Initially all blocks are available
        remaining_blocks = np.ones(n_blocks, dtype=bool)
        # For each triple, count how many remaining blocks cover it
        avail_counts = triple_counts.copy().astype(np.float64)

        while np.any(uncovered):
            if time.time() > deadline:
                break

            # Compute weights based on scarcity: rarer triples get higher weight
            if scarcity_mode == 'dynamic':
                # Weight = 1 / (number of remaining blocks covering this triple)
                # Only for uncovered triples
                w = np.zeros(n_triples, dtype=np.float64)
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) > 0:
                    w[unc_idx] = 1.0 / (avail_counts[unc_idx] + 1e-10)
            else:
                w = triple_weights

            # Compute gain for each block: sum of weights of uncovered triples it covers
            # Use the boolean matrix for speed
            gains = np.sum(block_contains * w[np.newaxis, :], axis=1)
            # Zero out gains for blocks that only cover already-covered triples
            gains[~remaining_blocks] = -1

            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                candidates = candidates[remaining_blocks[candidates]]
                if len(candidates) == 0:
                    break
                best_idx = candidates[rng.randrange(len(candidates))]
            else:
                valid_gains = gains.copy()
                valid_gains[~remaining_blocks] = -np.inf
                best_idx = int(np.argmax(valid_gains))
                if valid_gains[best_idx] <= 1e-12:
                    break

            selected.append(int(best_idx))

            # Update: mark this block as used, update uncovered and avail_counts
            remaining_blocks[best_idx] = False
            block_triples_best = block_triples[best_idx]
            uncovered[block_triples_best] = False

            # Update avail_counts: for each uncovered triple, decrement if this block covered it
            # A triple's avail_count decreases by 1 if the removed block covered it AND the triple is still uncovered
            for ti in block_triples_best:
                if uncovered[ti]:
                    avail_counts[ti] -= 1

        return selected

    def static_greedy_cover(random_tiebreak=False, weight_power=1.0, rng=None):
        if rng is None:
            rng = random.Random(42)
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        w = triple_weights ** weight_power if weight_power != 1.0 else triple_weights

        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = np.sum(uncovered[block_triples].astype(np.float64) * w[block_triples], axis=1)

            if random_tiebreak:
                max_gain = np.max(gains)
                if max_gain <= 1e-12:
                    break
                candidates = np.where(gains >= max_gain - 1e-12)[0]
                best_idx = candidates[rng.randrange(len(candidates))]
            else:
                best_idx = np.argmax(gains)
                if gains[best_idx] <= 1e-12:
                    break

            selected.append(int(best_idx))
            uncovered[block_triples[best_idx]] = False

        return selected

    def remove_redundant(block_indices):
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        result = list(block_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    return result
                bi = result[i]
                triples_in_block = block_triples[bi]
                if np.any(coverage[triples_in_block] <= 1):
                    continue
                coverage[triples_in_block] -= 1
                result.pop(i)
                changed = True
                break
        return result

    def local_search(block_indices, max_iterations=100):
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                block_triples_bi = block_triples[bi]

                if np.all(coverage[block_triples_bi] > 1):
                    coverage[block_triples_bi] -= 1
                    current.pop(i)
                    improved = True
                    break

                critical_mask = coverage[block_triples_bi] == 1
                if not np.any(critical_mask):
                    continue

                critical_triples = block_triples_bi[critical_mask]
                min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
                candidates = triple_to_blocks[min_ct]
                other_critical = critical_triples[critical_triples != min_ct]

                valid_candidates = []
                for cand in candidates:
                    cand = int(cand)
                    if cand == bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        cand_set = block_sets[cand]
                        all_covered = True
                        for ct in other_critical:
                            if triples[ct] not in cand_set:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(cand)

                if valid_candidates:
                    best_cand = valid_candidates[0]
                    coverage[block_triples_bi] -= 1
                    coverage[block_triples[best_cand]] += 1
                    current[i] = best_cand
                    improved = True
                    break

            if not improved:
                break
        return current

    def local_search_remove_two(block_indices, max_iterations=30):
        if len(block_indices) < 2:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    bi1 = current[i]
                    bi2 = current[j]
                    both_triples = np.union1d(block_triples[bi1], block_triples[bi2])
                    if np.all(coverage[both_triples] > 2):
                        coverage[both_triples] -= 2
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break
        return current

    def criticality_based_refinement(block_indices, max_iterations=200):
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False

            for i in range(len(current)):
                if time.time() > deadline:
                    break
                bi = current[i]
                bt = block_triples[bi]
                if np.all(coverage[bt] > 1):
                    coverage[bt] -= 1
                    current.pop(i)
                    improved = True
                    break

            if improved:
                continue

            min_crit = float('inf')
            min_idx = -1
            for i in range(len(current)):
                bi = current[i]
                bt = block_triples[bi]
                crit = np.sum(coverage[bt] == 1)
                if crit < min_crit:
                    min_crit = crit
                    min_idx = i

            if min_idx < 0 or min_crit == 0:
                break

            bi = current[min_idx]
            bt = block_triples[bi]
            critical_triples = bt[coverage[bt] == 1]
            if len(critical_triples) == 0:
                coverage[bt] -= 1
                current.pop(min_idx)
                improved = True
                continue

            min_ct = critical_triples[np.argmin(triple_counts[critical_triples])]
            candidates = triple_to_blocks[min_ct]
            other_critical = critical_triples[critical_triples != min_ct]

            best_cand = -1
            best_score = float('inf')

            for cand in candidates:
                cand = int(cand)
                if cand == bi:
                    continue
                if len(other_critical) == 0:
                    valid = True
                else:
                    cand_set = block_sets[cand]
                    valid = True
                    for ct in other_critical:
                        if triples[ct] not in cand_set:
                            valid = False
                            break
                if not valid:
                    continue

                cand_bt = block_triples[cand]
                overlap = np.intersect1d(bt, cand_bt, assume_unique=True)
                non_overlap = np.setdiff1d(bt, cand_bt, assume_unique=True)
                if len(non_overlap) > 0:
                    if np.any(coverage[non_overlap] <= 1):
                        continue

                score = -len(overlap)
                if score < best_score:
                    best_score = score
                    best_cand = cand

            if best_cand >= 0:
                coverage[bt] -= 1
                coverage[block_triples[best_cand]] += 1
                current[min_idx] = best_cand
                improved = True

            if not improved:
                break

        return current

    def two_exchange_search(block_indices, max_iterations=50):
        if len(block_indices) < 3:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)
        n = len(current)

        for _ in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False
            n = len(current)

            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break

                    bi1 = current[i]
                    bi2 = current[j]
                    bt1 = block_triples[bi1]
                    bt2 = block_triples[bi2]
                    both = np.union1d(bt1, bt2)

                    coverage[bt1] -= 1
                    coverage[bt2] -= 1

                    uncovered = np.where(coverage == 0)[0]

                    if len(uncovered) == 0:
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break

                    if len(uncovered) <= 2 * c_k_t:
                        gains = np.zeros(n_blocks, dtype=np.int32)
                        for ub in uncovered:
                            for blk in triple_to_blocks[ub]:
                                gains[blk] += 1

                        max_gain = np.max(gains)
                        if max_gain >= len(uncovered):
                            best_blk = int(np.argmax(gains))
                            if best_blk != bi1 and best_blk != bi2:
                                coverage[block_triples[best_blk]] += 1
                                current[i] = best_blk
                                current.pop(j)
                                improved = True
                                break
                        else:
                            top = np.argsort(gains)[::-1][:10]
                            found = False
                            for a in top:
                                if found:
                                    break
                                a = int(a)
                                if a == bi1 or a == bi2:
                                    continue
                                coverage[block_triples[a]] += 1
                                still_uncovered = np.where(coverage == 0)[0]
                                if len(still_uncovered) == 0:
                                    current[i] = a
                                    current.pop(j)
                                    improved = True
                                    found = True
                                    break
                                gains2 = np.zeros(n_blocks, dtype=np.int32)
                                for ub in still_uncovered:
                                    for blk in triple_to_blocks[ub]:
                                        gains2[blk] += 1
                                max_gain2 = np.max(gains2)
                                if max_gain2 >= len(still_uncovered):
                                    b = int(np.argmax(gains2))
                                    if b != bi1 and b != bi2 and b != a:
                                        coverage[block_triples[b]] += 1
                                        current[i] = a
                                        current[j] = b
                                        improved = True
                                        found = True
                                        break
                                coverage[block_triples[a]] -= 1

                            if found:
                                break

                    if not improved:
                        coverage[bt1] += 1
                        coverage[bt2] += 1

                if improved:
                    break

            if not improved:
                break

        return current

    def simulated_annealing(block_indices, time_budget=60.0):
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)
        best = list(current)
        best_size = len(current)

        rng = random.Random(777)
        T = 30.0
        T_min = 0.5
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 500

        sa_start = time.time()
        while time.time() - sa_start < time_budget and time.time() < deadline - 5.0:
            T *= cooling
            if T < T_min:
                T = T_min

            removed = False
            if len(current) > 0:
                indices = list(range(len(current)))
                rng.shuffle(indices)
                for i in indices:
                    bi = current[i]
                    if np.all(coverage[block_triples[bi]] > 1):
                        coverage[block_triples[bi]] -= 1
                        current.pop(i)
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                        removed = True
                        break

            if removed:
                continue

            if len(current) > 0:
                i = rng.randrange(len(current))
                old_bi = current[i]
                old_covered = block_triples[old_bi]

                critical = old_covered[coverage[old_covered] == 1]
                if len(critical) == 0:
                    coverage[old_covered] -= 1
                    current.pop(i)
                    if len(current) < best_size:
                        best_size = len(current)
                        best = list(current)
                        no_improve = 0
                    continue

                min_ct = critical[np.argmin(triple_counts[critical])]
                candidates = triple_to_blocks[min_ct]
                other_critical = critical[critical != min_ct]

                valid_candidates = []
                for cand in candidates:
                    cand = int(cand)
                    if cand == old_bi:
                        continue
                    if len(other_critical) == 0:
                        valid_candidates.append(cand)
                    else:
                        cand_set = block_sets[cand]
                        all_covered = True
                        for ct in other_critical:
                            if triples[ct] not in cand_set:
                                all_covered = False
                                break
                        if all_covered:
                            valid_candidates.append(cand)

                if not valid_candidates:
                    no_improve += 1
                    continue

                new_bi = valid_candidates[rng.randrange(len(valid_candidates))]
                new_covered = block_triples[new_bi]

                valid = True
                for idx in old_covered:
                    if coverage[idx] == 1 and idx not in new_covered:
                        valid = False
                        break

                if valid:
                    old_extra = float(np.sum(coverage[old_covered] > 1))
                    new_extra = float(np.sum(coverage[new_covered] > 1))
                    delta = new_extra - old_extra

                    if delta >= 0:
                        accept = True
                    else:
                        accept_prob = 1.0 / (1.0 + np.exp(delta / T))
                        accept = rng.random() < accept_prob

                    if accept:
                        coverage[old_covered] -= 1
                        coverage[new_covered] += 1
                        current[i] = new_bi
                        if len(current) < best_size:
                            best_size = len(current)
                            best = list(current)
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            if no_improve > max_no_improve:
                break

        return best

    def smooth_surrogate_refinement(block_indices, time_budget=20.0):
        """
        Refinement using a smooth surrogate: instead of hard min/max on criticality,
        use a soft-min (log-sum-exp) of coverage counts to guide block swaps.
        The idea: blocks that cover triples with low "soft-min coverage" are more valuable.
        """
        if len(block_indices) == 0:
            return block_indices

        coverage = np.zeros(n_triples, dtype=np.int32)
        for bi in block_indices:
            coverage[block_triples[bi]] += 1

        current = list(block_indices)
        best = list(current)
        best_size = len(current)

        rng = random.Random(1234)
        soft_beta = 5.0  # Temperature for log-sum-exp soft-min

        # Compute soft-min coverage for each triple:
        # soft_min(cov) = -log(sum(exp(-cov/beta))) / beta  (approximates min)
        # Lower soft_min means the triple is more "critically covered"

        def compute_block_value(bi, cov):
            """Compute a soft value for a block: sum of soft-scarcity of its triples."""
            bt = block_triples[bi]
            cov_bt = cov[bt].astype(np.float64)
            # Soft scarcity: 1 / (1 + soft_min) where soft_min approximates min coverage
            # Use: scarcity = beta / (1 + cov) as a smooth proxy for 1/min(cov)
            scarcity = soft_beta / (1.0 + cov_bt)
            return float(np.sum(scarcity))

        # Initial values
        values = np.array([compute_block_value(bi, coverage) for bi in current])

        start = time.time()
        while time.time() - start < time_budget and time.time() < deadline - 5.0:
            improved = False

            # Try removal of lowest-value block
            min_val_idx = int(np.argmin(values))
            bi = current[min_val_idx]
            bt = block_triples[bi]
            if np.all(coverage[bt] > 1):
                coverage[bt] -= 1
                current.pop(min_val_idx)
                values = np.delete(values, min_val_idx)
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
                improved = True
                continue

            # Try swap: find block with lowest value, try to replace with higher-value block
            i = int(np.argmin(values))
            bi = current[i]
            bt = block_triples[bi]
            critical = bt[coverage[bt] == 1]

            if len(critical) == 0:
                coverage[bt] -= 1
                current.pop(i)
                values = np.delete(values, i)
                if len(current) < best_size:
                    best_size = len(current)
                    best = list(current)
                improved = True
                continue

            # Find candidates that cover all critical triples
            min_ct = critical[np.argmin(triple_counts[critical])]
            candidates = triple_to_blocks[min_ct]
            other_critical = critical[critical != min_ct]

            best_cand = -1
            best_val_gain = -float('inf')

            for cand in candidates:
                cand = int(cand)
                if cand == bi:
                    continue
                if len(other_critical) > 0:
                    valid = True
                    for ct in other_critical:
                        if not block_contains[cand, ct]:
                            valid = False
                            break
                    if not valid:
                        continue

                # Check swap validity
                non_overlap = np.setdiff1d(bt, block_triples[cand], assume_unique=True)
                if len(non_overlap) > 0:
                    if np.any(coverage[non_overlap] <= 1):
                        continue

                # Compute value gain
                new_val = compute_block_value(cand, coverage)
                gain = new_val - values[i]
                if gain > best_val_gain:
                    best_val_gain = gain
                    best_cand = cand

            if best_cand >= 0 and best_val_gain > -1e-6:
                coverage[bt] -= 1
                coverage[block_triples[best_cand]] += 1
                current[i] = best_cand
                values[i] = compute_block_value(best_cand, coverage)
                improved = True

            if not improved:
                # Perturb: randomly swap two blocks to escape local optima
                if len(current) >= 2 and rng.random() < 0.3:
                    i1 = rng.randrange(len(current))
                    i2 = rng.randrange(len(current))
                    if i1 != i2:
                        current[i1], current[i2] = current[i2], current[i1]
                        values[i1], values[i2] = values[i2], values[i1]
                else:
                    break

        return best

    # Phase 1: Deterministic greedy with different strategies
    best_solution = None
    best_size = float('inf')

    # Dynamic scarcity greedy (the key improvement)
    for mode in ['dynamic', 'static']:
        if time.time() > deadline - 25.0:
            break
        if mode == 'dynamic':
            selected = dynamic_greedy_cover(random_tiebreak=False, scarcity_mode='dynamic')
        else:
            selected = static_greedy_cover(random_tiebreak=False, weight_power=1.5)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=30)
        selected = criticality_based_refinement(selected, max_iterations=50)
        selected = local_search_remove_two(selected, max_iterations=10)

        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]

    # Also try static with different weight powers
    for wp in [1.0, 2.0]:
        if time.time() > deadline - 20.0:
            break
        selected = static_greedy_cover(random_tiebreak=False, weight_power=wp)
        selected = remove_redundant(selected)
        selected = local_search(selected, max_iterations=20)
        selected = criticality_based_refinement(selected, max_iterations=40)
        selected = local_search_remove_two(selected, max_iterations=10)

        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(all_blocks[i]) for i in selected]

    # Phase 2: Random restarts with SA refinement
    np.random.seed(42)
    random.seed(42)

    elapsed = time.time() - start_time
    remaining = deadline - 5.0 - elapsed
    if remaining > 5.0:
        num_restarts = max(1, int(remaining / 4.0))
        num_restarts = min(num_restarts, 20)

        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                break

            # Mix dynamic and static greedy with random tiebreak
            if r % 2 == 0:
                selected = dynamic_greedy_cover(random_tiebreak=True, rng=random.Random(42 + r), scarcity_mode='dynamic')
            else:
                wp = random.choice([1.0, 1.5, 2.0])
                selected = static_greedy_cover(random_tiebreak=True, weight_power=wp, rng=random.Random(42 + r))

            selected = remove_redundant(selected)
            selected = local_search(selected, max_iterations=15)
            selected = criticality_based_refinement(selected, max_iterations=30)

            sa_time = min(10.0, max(3.0, (deadline - 10.0 - time.time()) * 0.3))
            selected = simulated_annealing(selected, time_budget=sa_time)

            if len(selected) < best_size:
                best_size = len(selected)
                best_solution = [list(all_blocks[i]) for i in selected]

    # Phase 3: Final polish on best solution
    if best_solution is not None:
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_solution]

        best_indices = criticality_based_refinement(best_indices, max_iterations=80)
        best_indices = local_search(best_indices, max_iterations=40)
        best_indices = remove_redundant(best_indices)
        best_indices = local_search_remove_two(best_indices, max_iterations=10)

        # Smooth surrogate refinement
        remaining = deadline - 5.0 - time.time()
        if remaining > 5.0:
            best_indices = smooth_surrogate_refinement(best_indices, time_budget=min(10.0, remaining - 3.0))

        # 2-exchange search
        remaining = deadline - 5.0 - time.time()
        if remaining > 3.0:
            best_indices = two_exchange_search(best_indices, max_iterations=20)

        # Final SA polish
        remaining = deadline - 5.0 - time.time()
        if remaining > 3.0:
            best_indices = simulated_annealing(best_indices, time_budget=min(10.0, remaining - 2.0))

        # Final criticality pass
        best_indices = criticality_based_refinement(best_indices, max_iterations=40)
        best_indices = remove_redundant(best_indices)

        best_solution = [list(all_blocks[i]) for i in best_indices]
        best_size = len(best_solution)

    return best_solution
