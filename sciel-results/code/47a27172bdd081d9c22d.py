import itertools
import time
import random
import numpy as np
from functools import lru_cache

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 118.0  # Safety margin for 150s limit

    points = list(range(v))

    # Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # Build incidence matrix: block x t-subset, 1 if t-subset is in block
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute block gain (number of t-subsets covered)
    block_tsub_counts = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)

    # Bitset representation for exact cover solver
    # Each block is represented as a bitset of t-subsets it covers
    # N_t can be up to C(19,5) = 11628, so we use Python integers as bitsets
    block_bitsets = []
    for subs in block_tsubs:
        bits = 0
        for si in subs:
            bits |= (1 << si)
        block_bitsets.append(bits)

    def run_greedy(seed):
        random.seed(seed)
        block_gain = block_tsub_counts.copy()
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []

        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = int(random.choice(candidates.tolist()))

            selected_indices.append(best_block_idx)

            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    for b_idx in tsub_blocks[si]:
                        block_gain[b_idx] -= 1

        return selected_indices

    def exact_cover_search(uncovered_tsubs, available_blocks, time_limit=5.0):
        """
        Branch-and-bound exact cover solver using bitsets.
        Returns a list of block indices that cover all uncovered_tsubs, or None.
        """
        if not uncovered_tsubs:
            return []

        start = time.time()
        best_solution = None
        best_size = len(available_blocks) + 1  # Upper bound

        # Convert to bitset
        uncovered_bits = 0
        for si in uncovered_tsubs:
            uncovered_bits |= (1 << si)

        def search(uncovered_bits, available_blocks_set, depth, current_solution):
            nonlocal best_solution, best_size

            if time.time() - start > time_limit:
                return None

            if uncovered_bits == 0:
                if len(current_solution) < best_size:
                    best_solution = list(current_solution)
                    best_size = len(current_solution)
                return current_solution

            # Lower bound: we need at least ceil(popcount(uncovered_bits) / max_coverage) more blocks
            # But simpler: if we've already exceeded best_size, prune
            if depth >= best_size:
                return None

            # Find the t-subset with fewest available blocks (MRV heuristic)
            # Iterate through uncovered t-subsets and count available blocks
            min_count = float('inf')
            min_tsub = -1
            min_blocks = None

            temp_bits = uncovered_bits
            while temp_bits:
                # Get lowest set bit
                lowbit = temp_bits & (-temp_bits)
                si = lowbit.bit_length() - 1
                temp_bits ^= lowbit

                # Count available blocks that cover this t-subset
                count = 0
                blocks_covering = []
                for b in available_blocks_set:
                    if (block_bitsets[b] & lowbit):
                        count += 1
                        blocks_covering.append(b)
                        if count >= min_count:
                            break

                if count < min_count:
                    min_count = count
                    min_tsub = si
                    min_blocks = blocks_covering
                    if count <= 1:
                        break

            if min_tsub == -1 or min_count == 0:
                return None

            # Try each block that covers min_tsub
            for b in min_blocks:
                # Remove b from available
                new_available = available_blocks_set - {b}

                # Update uncovered: remove all t-subsets covered by b
                new_uncovered = uncovered_bits & ~block_bitsets[b]

                result = search(new_uncovered, new_available, depth + 1, current_solution + [b])
                if result is not None:
                    return result

            return None

        # Start search
        result = search(uncovered_bits, set(available_blocks), 0, [])
        if result is not None:
            return result
        return None

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)
        selected_set = set(selected_indices)

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        for i, g in enumerate(selected_indices):
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel

        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue

            removed[i] = True
            g = selected_indices[i]

            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]

                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        if g_other in selected_set:
                            for j in range(n_sel):
                                if selected_indices[j] == g_other and not removed[j]:
                                    crit_count[j] += 1
                                    break

        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def polish_with_exact_cover(selected_indices, time_budget=8.0):
        """
        Try to improve solution by finding exact covers for subsets of uncovered t-subsets.
        Strategy: for each block, try removing it and finding a smaller replacement set.
        """
        if not selected_indices or time.time() > DEADLINE - 3.0:
            return selected_indices

        start = time.time()
        sol = list(selected_indices)
        sol_set = set(sol)

        # Compute coverage
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in sol:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        while improved and time.time() < start + time_budget and time.time() < DEADLINE - 3.0:
            improved = False
            for i in range(len(sol)):
                if time.time() > DEADLINE - 3.0:
                    break
                g = sol[i]

                # Check if removing g would uncover any t-subsets
                would_uncover = False
                uncovered_after_removal = []
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        would_uncover = True
                        uncovered_after_removal.append(si)

                if not would_uncover:
                    # Safe to remove
                    for si in block_tsubs[g]:
                        total_count[si] -= 1
                    sol.pop(i)
                    sol_set.discard(g)
                    improved = True
                    break

                # Try to find a smaller replacement
                # Use exact cover search with limited available blocks
                # First, temporarily remove g
                for si in block_tsubs[g]:
                    total_count[si] -= 1

                # Find blocks that cover at least one of the uncovered t-subsets
                candidate_blocks = set()
                for si in uncovered_after_removal:
                    for b in tsub_blocks[si]:
                        if b not in sol_set:
                            candidate_blocks.add(b)

                if len(candidate_blocks) <= 50:  # Limit search space
                    replacement = exact_cover_search(
                        uncovered_after_removal,
                        list(candidate_blocks),
                        time_limit=min(2.0, DEADLINE - 3.0 - time.time())
                    )
                    if replacement is not None and len(replacement) < 1:
                        # This shouldn't happen since we need to cover something
                        pass
                    elif replacement is not None and len(replacement) <= 1:
                        # Found a smaller or equal replacement
                        for si in block_tsubs[g]:
                            total_count[si] += 1  # Revert
                        sol.pop(i)
                        sol_set.discard(g)
                        for b in replacement:
                            sol.append(b)
                            sol_set.add(b)
                            for si in block_tsubs[b]:
                                total_count[si] += 1
                        improved = True
                        break
                    else:
                        # Revert
                        for si in block_tsubs[g]:
                            total_count[si] += 1
                else:
                    # Too many candidates, just revert
                    for si in block_tsubs[g]:
                        total_count[si] += 1

        return sol

    def local_search(selected_indices, max_iters=50):
        """Try to remove a block and add fewer blocks to re-cover uncovered t-subsets."""
        if not selected_indices:
            return selected_indices

        n_sel = len(selected_indices)
        selected_set = set(selected_indices)
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            for i in range(n_sel):
                if time.time() > DEADLINE:
                    break
                g = selected_indices[i]
                would_uncover = False
                for si in block_tsubs[g]:
                    if total_count[si] <= 1:
                        would_uncover = True
                        break
                if would_uncover:
                    continue

                for si in block_tsubs[g]:
                    total_count[si] -= 1

                uncovered = [si for si in block_tsubs[g] if total_count[si] == 0]

                if not uncovered:
                    selected_indices.pop(i)
                    selected_set.discard(g)
                    n_sel -= 1
                    improved = True
                    break

                found_one = False
                for g_new in all_blocks:
                    if g_new in selected_set:
                        continue
                    covers_all = all(si in block_tsubs[g_new] for si in uncovered)
                    if covers_all:
                        found_one = True
                        for si in block_tsubs[g_new]:
                            total_count[si] += 1
                        selected_indices.append(g_new)
                        selected_set.add(g_new)
                        n_sel += 1
                        selected_indices.pop(i)
                        selected_set.discard(g)
                        n_sel -= 1
                        improved = True
                        break

                if not found_one:
                    for si in block_tsubs[g]:
                        total_count[si] += 1
                    continue

        return selected_indices

    def simulated_annealing(selected_indices, time_budget=20.0):
        """Simulated annealing: state is a set of block indices.
        Energy = uncovered_count + lambda * len(sol).
        Move: swap one block with another, or remove/add a block.
        Accept worse moves with probability exp(-delta_E / T)."""
        if not selected_indices or len(selected_indices) < 2:
            return selected_indices

        sa_start = time.time()
        sa_deadline = sa_start + time_budget

        # Initial state
        sol = list(selected_indices)
        sol_set = set(sol)

        # Compute initial coverage counts
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in sol:
            for si in block_tsubs[g]:
                total_count[si] += 1
        uncovered = int(np.sum(total_count == 0))

        # Energy function: uncovered + penalty * len(sol)
        # We want to minimize both uncovered and block count
        # Use a weight that makes uncovered dominate
        penalty_weight = 5.0  # Each uncovered t-subset costs 5x a block

        def energy(uncovered_count, sol_size):
            return uncovered_count + penalty_weight * sol_size

        best_sol = list(sol)
        best_uncovered = uncovered
        best_energy = energy(uncovered, len(sol))

        # Temperature schedule
        T0 = 10.0
        T_min = 0.01
        # We'll run for a fixed number of steps, cooling geometrically
        # Estimate steps based on time budget
        n_steps = 5000

        T = T0
        cooling = (T_min / T0) ** (1.0 / n_steps)

        current_energy = energy(uncovered, len(sol))

        for step in range(n_steps):
            if time.time() > sa_deadline:
                break

            T *= cooling

            # Generate a move: pick a random block in sol, replace with random block
            if len(sol) > 0:
                idx = random.randint(0, len(sol) - 1)
                old_block = sol[idx]

                # Try swapping with a random other block
                new_block = random.randint(0, N_blocks - 1)
                if new_block == old_block:
                    continue

                # Compute delta energy
                # Remove old_block: update total_count
                delta_uncovered = 0
                # First, count how many t-subsets become uncovered when removing old_block
                # and how many are already covered by other blocks
                newly_uncovered = 0
                for si in block_tsubs[old_block]:
                    if total_count[si] == 1:
                        newly_uncovered += 1

                # Now add new_block: count how many of those newly_uncovered get covered
                # and how many additional t-subsets get covered (already covered ones don't matter)
                # Actually, let's compute delta properly:
                # delta_uncovered = (newly_uncovered from removal) - (covered by new_block among those)
                # But we also need to account for t-subsets that were already uncovered and get covered by new_block

                # More carefully:
                # After removal: uncovered increases by newly_uncovered
                # After addition: uncovered decreases by (number of currently uncovered t-subsets in new_block)
                # Currently uncovered = original uncovered + newly_uncovered
                # Of the t-subsets in new_block, some are already covered, some were uncovered

                # Let's just compute it directly
                # Temporarily remove old_block
                for si in block_tsubs[old_block]:
                    total_count[si] -= 1

                # Count newly uncovered
                temp_uncovered = 0
                for si in block_tsubs[old_block]:
                    if total_count[si] == 0:
                        temp_uncovered += 1

                # Add new_block
                for si in block_tsubs[new_block]:
                    total_count[si] += 1

                # Count how many of the previously uncovered are now covered
                # The total uncovered changed by: temp_uncovered (from removal) - (covered by new_block among previously uncovered)
                # Actually, let's just count the new uncovered count
                new_uncovered = 0
                for si in block_tsubs[old_block] + block_tsubs[new_block]:
                    if total_count[si] == 0:
                        new_uncovered += 1
                        # But we might double count if a t-subset is in both blocks
                        # Let's use a set approach for correctness

                # This is getting complicated. Let's use a simpler approach:
                # Just compute the full uncovered count after the move
                # But that's O(N_t) which is expensive.

                # Alternative: track delta more carefully
                # Let's revert and use a different strategy

                # Revert the changes
                for si in block_tsubs[new_block]:
                    total_count[si] -= 1
                for si in block_tsubs[old_block]:
                    total_count[si] += 1

                # Simpler approach: just do the move and compute delta by tracking
                # how many t-subsets change coverage status

                # Remove old_block
                removed_uncovered = 0
                for si in block_tsubs[old_block]:
                    if total_count[si] == 1:
                        removed_uncovered += 1
                    total_count[si] -= 1

                # Add new_block
                added_covered = 0
                for si in block_tsubs[new_block]:
                    if total_count[si] == 0:
                        added_covered += 1
                    total_count[si] += 1

                # Net change in uncovered: removed_uncovered - added_covered
                delta_uncovered = removed_uncovered - added_covered
                delta_size = 0  # Size stays the same in a swap

                delta_E = (delta_uncovered) + penalty_weight * delta_size

                # Accept or reject
                if delta_E <= 0 or random.random() < np.exp(-delta_E / max(T, 1e-10)):
                    # Accept the move
                    sol[idx] = new_block
                    sol_set.discard(old_block)
                    sol_set.add(new_block)
                    uncovered += delta_uncovered
                    current_energy += delta_E

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                        best_uncovered = uncovered
                else:
                    # Reject: revert
                    for si in block_tsubs[new_block]:
                        total_count[si] -= 1
                    for si in block_tsubs[old_block]:
                        total_count[si] += 1

            # Occasionally try adding or removing a block
            if random.random() < 0.1 and len(sol) > 0:
                # Try removing a block
                idx = random.randint(0, len(sol) - 1)
                old_block = sol[idx]

                removed_uncovered = 0
                for si in block_tsubs[old_block]:
                    if total_count[si] == 1:
                        removed_uncovered += 1

                if removed_uncovered == 0:
                    # Safe to remove
                    for si in block_tsubs[old_block]:
                        total_count[si] -= 1
                    sol.pop(idx)
                    sol_set.discard(old_block)
                    uncovered += 0  # No change in uncovered
                    current_energy -= penalty_weight

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_sol = list(sol)
                        best_uncovered = uncovered
                else:
                    # Don't remove if it would uncover t-subsets
                    pass

            elif random.random() < 0.1 and len(sol) < N_blocks:
                # Try adding a block
                new_block = random.randint(0, N_blocks - 1)
                if new_block not in sol_set:
                    added_covered = 0
                    for si in block_tsubs[new_block]:
                        if total_count[si] == 0:
                            added_covered += 1

                    delta_E = -added_covered + penalty_weight

                    if delta_E <= 0 or random.random() < np.exp(-delta_E / max(T, 1e-10)):
                        for si in block_tsubs[new_block]:
                            total_count[si] += 1
                        sol.append(new_block)
                        sol_set.add(new_block)
                        uncovered -= added_covered
                        current_energy += delta_E

                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_sol = list(sol)
                            best_uncovered = uncovered

        # Return the best solution found
        # Make sure it's valid (no duplicates)
        best_unique = list(set(best_sol))
        return best_unique

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)

    # Randomized restarts with simulated annealing improvement
    while time.time() < DEADLINE - 10.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
            if time.time() < DEADLINE - 15.0:
                improved = simulated_annealing(list(cand_idx), time_budget=15.0)
                improved = reduce_solution_fast(improved)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_sol = [all_blocks[i] for i in improved]

    # Final local search pass on the best solution
    if time.time() < DEADLINE - 5.0:
        final_idx = [all_blocks.index(b) for b in best_sol]
        improved = local_search(list(final_idx), max_iters=30)
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol = [all_blocks[i] for i in improved]

    # Final polish with exact cover search
    if time.time() < DEADLINE - 2.0:
        final_idx = [all_blocks.index(b) for b in best_sol]
        polished = polish_with_exact_cover(list(final_idx), time_budget=5.0)
        polished = reduce_solution_fast(polished)
        if len(polished) < best_len:
            best_len = len(polished)
            best_sol = [all_blocks[i] for i in polished]

    return best_sol
