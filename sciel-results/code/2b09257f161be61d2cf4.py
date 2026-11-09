import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute: for each block, which t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # For each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Convert to numpy arrays
    block_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_tsubs]
    tsub_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_blocks]

    # Precompute a bitset representation for each block's t-subsets
    # Each block covers C(8,5)=56 t-subsets. We'll use a set for fast intersection.
    block_tsets = [set(subs) for subs in block_tsubs]

    # Precompute t-subset to block list as sets for faster operations
    tsub_blocksets = [set(lst) for lst in tsub_blocks]

    def run_greedy_indices(randomize=False, seed=None, alpha=1.0):
        """
        Greedy with optional randomization. alpha controls how much we weight
        previously selected blocks' coverage (to encourage diversity).
        """
        if seed is not None:
            random.seed(seed)

        # block_gain[i] = number of uncovered t-subsets covered by block i
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []

        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = int(np.max(block_gain))
            if max_gain <= 0:
                break

            if randomize:
                candidates = np.where(block_gain == max_gain)[0]
                best_block_idx = random.choice(candidates.tolist())
            else:
                best_block_idx = int(np.argmax(block_gain))

            selected_indices.append(best_block_idx)

            subs_np = block_tsubs_np[best_block_idx]
            newly_covered_mask = is_covered[subs_np] == 0
            newly_covered = subs_np[newly_covered_mask]

            if len(newly_covered) > 0:
                is_covered[newly_covered] = 1
                uncovered_count -= len(newly_covered)

                # Update gains for all blocks that share t-subsets with newly covered ones
                all_block_indices = []
                for si in newly_covered:
                    all_block_indices.extend(tsub_blocks[si])
                if all_block_indices:
                    all_block_indices_np = np.array(all_block_indices, dtype=np.int32)
                    np.subtract.at(block_gain, all_block_indices_np, 1)

        return selected_indices

    def local_search_reduce(selected, time_limit=5.0):
        """
        Intensive local search using multiple strategies.
        """
        if not selected:
            return selected

        start = time.time()
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1

        max_iterations = 8000

        for iteration in range(max_iterations):
            if time.time() - start > time_limit or time.time() > DEADLINE:
                break

            improved = False

            # Strategy 1: Remove redundant blocks (try more)
            for _ in range(8):
                b_idx = random.choice(selected)
                subs_np = block_tsubs_np[b_idx]
                if int(np.sum(cov[subs_np] == 1)) == 0:
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    improved = True
                    break

            if improved:
                continue

            # Strategy 2: 1-for-1 swap
            if len(selected) >= 1:
                b_remove = random.choice(selected)
                subs_remove = block_tsubs_np[b_remove]
                would_uncover_mask = cov[subs_remove] == 1
                would_uncover_count = int(np.sum(would_uncover_mask))

                if would_uncover_count > 0:
                    would_uncover = subs_remove[would_uncover_mask]

                    # Find candidate blocks that cover most of would_uncover
                    candidate_set = set()
                    for si in would_uncover[:20]:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b_remove)
                    candidate_set -= selected_set

                    if candidate_set:
                        candidates = list(candidate_set)
                        sample = candidates if len(candidates) <= 15 else random.sample(candidates, 15)
                        best_b_add = -1
                        best_score = -1

                        for b_add in sample:
                            subs_add = block_tsubs_np[b_add]
                            covered = int(np.sum(np.isin(subs_add, would_uncover)))
                            extra = int(np.sum(cov[subs_add] == 0))
                            score = covered + extra * 0.05
                            if score > best_score:
                                best_score = score
                                best_b_add = b_add

                        if best_b_add != -1 and best_score >= would_uncover_count:
                            subs_add = block_tsubs_np[best_b_add]
                            selected.remove(b_remove)
                            selected_set.discard(b_remove)
                            cov[subs_remove] -= 1

                            selected.append(best_b_add)
                            selected_set.add(best_b_add)
                            cov[subs_add] += 1
                            improved = True

            if improved:
                continue

            # Strategy 3: 2-for-1 exchange
            if len(selected) >= 2 and time.time() - start < time_limit * 0.5:
                b1, b2 = random.sample(selected, 2)
                subs1 = block_tsubs_np[b1]
                subs2 = block_tsubs_np[b2]

                cov_temp = cov.copy()
                cov_temp[subs1] -= 1
                cov_temp[subs2] -= 1
                would_uncover_mask = cov_temp == 0
                would_uncover_count = int(np.sum(would_uncover_mask))

                if would_uncover_count > 0:
                    would_uncover = np.where(would_uncover_mask)[0]

                    candidate_set = set()
                    for si in would_uncover[:30]:
                        candidate_set.update(tsub_blocks[si])
                    candidate_set.discard(b1)
                    candidate_set.discard(b2)
                    candidate_set -= selected_set

                    if candidate_set:
                        candidates = list(candidate_set)
                        sample = candidates if len(candidates) <= 20 else random.sample(candidates, 20)
                        best_b_add = -1
                        best_coverage = -1

                        for b_add in sample:
                            subs_add = block_tsubs_np[b_add]
                            coverage = int(np.sum(np.isin(subs_add, would_uncover)))
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_b_add = b_add

                        if best_b_add != -1 and best_coverage >= would_uncover_count:
                            subs_add = block_tsubs_np[best_b_add]
                            selected.remove(b1)
                            selected_set.discard(b1)
                            selected.remove(b2)
                            selected_set.discard(b2)
                            cov[subs1] -= 1
                            cov[subs2] -= 1

                            selected.append(best_b_add)
                            selected_set.add(best_b_add)
                            cov[subs_add] += 1
                            improved = True

            if not improved:
                # Try a few more random removals
                for _ in range(5):
                    if not selected:
                        break
                    b_idx = random.choice(selected)
                    subs_np = block_tsubs_np[b_idx]
                    if int(np.sum(cov[subs_np] == 1)) == 0:
                        selected.remove(b_idx)
                        selected_set.discard(b_idx)
                        cov[subs_np] -= 1
                        improved = True
                        break
                if not improved:
                    break

        return selected

    def simulated_anneal(initial_indices, time_limit=8.0):
        """
        Simulated Annealing with efficient move generation.
        """
        if not initial_indices:
            return initial_indices

        M = 500
        T = 5.0
        alpha = 0.98
        T_min = 0.01

        selected = list(initial_indices)
        selected_set = set(selected)
        cov = np.zeros(N_t, dtype=np.int32)
        for b_idx in selected:
            cov[block_tsubs_np[b_idx]] += 1

        uncovered_count = int(np.sum(cov == 0))
        current_cost = len(selected) + M * uncovered_count
        best_cost = current_cost
        best_selected = list(selected)

        start = time.time()

        while T > T_min and (time.time() - start) < time_limit and time.time() < DEADLINE:
            move_type = random.random()

            if move_type < 0.35 and selected:
                # Remove a block
                b_idx = random.choice(selected)
                subs_np = block_tsubs_np[b_idx]
                delta_uncovered = int(np.sum(cov[subs_np] == 1))
                delta_cost = -1 + M * delta_uncovered

                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected.remove(b_idx)
                    selected_set.discard(b_idx)
                    cov[subs_np] -= 1
                    uncovered_count += delta_uncovered
                    current_cost += delta_cost

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = list(selected)

            elif move_type < 0.60:
                # Add a block covering uncovered t-subsets
                if uncovered_count > 0:
                    uncovered_mask = (cov == 0).nonzero()[0]
                    if len(uncovered_mask) > 0:
                        target_si = int(uncovered_mask[random.randint(0, len(uncovered_mask) - 1)])
                        candidate_blocks = tsub_blocks[target_si]
                        if candidate_blocks:
                            b_idx = random.choice(candidate_blocks)
                            if b_idx not in selected_set:
                                subs_np = block_tsubs_np[b_idx]
                                newly_covered = int(np.sum(cov[subs_np] == 0))
                                delta_cost = 1 - M * newly_covered

                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    selected.append(b_idx)
                                    selected_set.add(b_idx)
                                    cov[subs_np] += 1
                                    uncovered_count -= newly_covered
                                    current_cost += delta_cost

                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)
                else:
                    b_idx = random.randint(0, N_blocks - 1)
                    if b_idx not in selected_set:
                        delta_cost = 1
                        if random.random() < np.exp(-delta_cost / T):
                            selected.append(b_idx)
                            selected_set.add(b_idx)
                            cov[block_tsubs_np[b_idx]] += 1
                            current_cost += delta_cost

            else:
                # Swap: remove one, add another
                if len(selected) >= 1:
                    b_remove = random.choice(selected)
                    subs_remove = block_tsubs_np[b_remove]
                    would_uncover_mask = cov[subs_remove] == 1
                    would_uncover_count = int(np.sum(would_uncover_mask))

                    if would_uncover_count > 0:
                        would_uncover = subs_remove[would_uncover_mask]
                        target_si = int(random.choice(would_uncover))
                        candidate_blocks = tsub_blocks[target_si]
                        candidates = [b for b in candidate_blocks if b != b_remove and b not in selected_set]
                        if not candidates:
                            candidates = [b for b in candidate_blocks if b != b_remove]
                        if candidates:
                            sample = candidates if len(candidates) <= 10 else random.sample(candidates, 10)
                            best_b_add = -1
                            best_score = -1

                            for b_add in sample:
                                subs_add = block_tsubs_np[b_add]
                                score = int(np.sum(np.isin(subs_add, would_uncover)))
                                if score > best_score:
                                    best_score = score
                                    best_b_add = b_add

                            if best_b_add != -1:
                                subs_add = block_tsubs_np[best_b_add]
                                delta_cover_add = int(np.sum((cov[subs_add] == 0) | (np.isin(subs_add, would_uncover) & (cov[subs_add] == 1))))
                                net_uncover_change = would_uncover_count - delta_cover_add
                                delta_cost = net_uncover_change * M

                                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                                    selected.remove(b_remove)
                                    selected_set.discard(b_remove)
                                    cov[subs_remove] -= 1
                                    uncovered_count += would_uncover_count

                                    selected.append(best_b_add)
                                    selected_set.add(best_b_add)
                                    cov[subs_add] += 1
                                    uncovered_count -= delta_cover_add
                                    current_cost += delta_cost

                                    if current_cost < best_cost:
                                        best_cost = current_cost
                                        best_selected = list(selected)

            T *= alpha

        # Repair
        uncovered_mask = (cov == 0).nonzero()[0]
        while len(uncovered_mask) > 0:
            best_b = -1
            max_cov = 0
            candidate_blocks = set()
            for si in uncovered_mask[:25]:
                candidate_blocks.update(tsub_blocks[si])

            for b_idx in candidate_blocks:
                if b_idx in selected_set:
                    continue
                count = int(np.sum(cov[block_tsubs_np[b_idx]] == 0))
                if count > max_cov:
                    max_cov = count
                    best_b = b_idx

            if best_b == -1:
                break

            selected.append(best_b)
            selected_set.add(best_b)
            cov[block_tsubs_np[best_b]] += 1
            uncovered_mask = (cov == 0).nonzero()[0]

        # Final reduction
        remaining = min(3.0, DEADLINE - time.time() - 1.5)
        if remaining > 0.5:
            selected = local_search_reduce(selected, time_limit=remaining)

        return selected

    def vectorized_greedy():
        """
        Fully vectorized greedy: at each step, compute the gain for all blocks
        simultaneously using numpy.
        """
        # coverage_count[s] = number of selected blocks covering t-subset s
        coverage_count = np.zeros(N_t, dtype=np.int32)
        selected_indices = []

        # Precompute block_tsubs as a list of arrays (already done)
        # We need to compute, for each block, how many uncovered t-subsets it covers.

        # Efficient approach: maintain an array 'uncovered' of size N_t (bool),
        # and for each block, gain = sum(uncovered[block_tsubs[i]])

        uncovered = np.ones(N_t, dtype=np.int8)
        uncovered_count = N_t

        while uncovered_count > 0 and time.time() < DEADLINE:
            # Compute gain for all blocks: gain[i] = sum(uncovered[block_tsubs[i]])
            # This is expensive to do naively. Use a different approach:
            # For each uncovered t-subset, it contributes 1 to the gain of each block covering it.
            # So gain[i] = sum over uncovered s of (1 if s in block_tsubs[i] else 0)

            # We can compute this as: for each uncovered s, add 1 to all blocks in tsub_blocks[s]
            # But that's O(N_t * avg_blocks_per_tsub) per iteration.

            # Alternative: precompute a matrix M where M[i,s] = 1 if block i covers t-subset s
            # Then gain = M @ uncovered. But M is N_blocks x N_t = 75582 x 11628, too large.

            # Use the fact that each block covers only 56 t-subsets.
            # gain[i] = sum(uncovered[block_tsubs[i]])
            # We can compute this in a loop over blocks, but vectorize the inner sum.

            gains = np.zeros(N_blocks, dtype=np.int32)
            for i in range(N_blocks):
                gains[i] = int(np.sum(uncovered[block_tsubs_np[i]]))

            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                break

            selected_indices.append(best_idx)
            subs = block_tsubs_np[best_idx]
            newly = uncovered[subs] == 1
            newly_covered = subs[newly]
            if len(newly_covered) > 0:
                uncovered[newly_covered] = 0
                uncovered_count -= len(newly_covered)

        return selected_indices

    # Main loop
    best_sol_idx = None
    best_len = float('inf')

    # 1. Deterministic greedy
    sol_idx = run_greedy_indices(randomize=False)
    sol_idx = simulated_anneal(sol_idx, time_limit=10.0)
    best_sol_idx = sol_idx
    best_len = len(sol_idx)

    # 2. Randomized restarts
    restart_count = 0
    while time.time() < DEADLINE - 15:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy_indices(randomize=True, seed=seed)
        remaining = DEADLINE - time.time()
        if remaining < 12:
            break
        time_limit = min(6.0, remaining - 12.0)
        if time_limit < 1.0:
            break

        cand_idx = simulated_anneal(cand_idx, time_limit=time_limit)

        if len(cand_idx) < best_len:
            best_sol_idx = cand_idx
            best_len = len(cand_idx)

        restart_count += 1

    # 3. Final polish
    if best_sol_idx is not None and time.time() < DEADLINE - 2:
        selected = list(best_sol_idx)
        polish_time = min(8.0, DEADLINE - time.time() - 1.5)
        if polish_time > 1.0:
            selected = local_search_reduce(selected, time_limit=polish_time * 0.4)

            if time.time() < DEADLINE - 2:
                remaining = DEADLINE - time.time() - 2.0
                if remaining > 2.0:
                    polished = simulated_anneal(selected, time_limit=min(4.0, remaining))
                    if len(polished) < len(selected):
                        selected = polished

            if time.time() < DEADLINE - 1:
                remaining = DEADLINE - time.time() - 1.0
                if remaining > 0.5:
                    selected = local_search_reduce(selected, time_limit=remaining)

            if len(selected) < best_len:
                best_sol_idx = selected
                best_len = len(selected)

    return [all_blocks[i] for i in best_sol_idx]
