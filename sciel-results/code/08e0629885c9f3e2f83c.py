import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 150s limit

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

    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute block t-subset membership as sets for fast lookup
    block_tsubs_set = [set(subs) for subs in block_tsubs]

    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
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
                    block_gain[tsub_blocks[si]] -= 1

        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)
        local_of_global = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            local_of_global[g] = i

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = np.zeros(n_sel, dtype=np.bool_)

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
                        i_other = local_of_global[g_other]
                        if i_other >= 0 and not removed[i_other]:
                            crit_count[i_other] += 1

        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search(selected_indices, max_iters=50):
        """1-for-1 swap local search: try replacing a critical block with a better one."""
        if not selected_indices:
            return selected_indices

        n_sel = len(selected_indices)
        in_sol = np.zeros(N_blocks, dtype=np.bool_)
        for g in selected_indices:
            in_sol[g] = True
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            n_sel = len(selected_indices)
            for i in range(n_sel):
                if time.time() > DEADLINE - 2.0:
                    return selected_indices
                g = selected_indices[i]
                # Find t-subsets uniquely covered by this block
                unique_si = []
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        unique_si.append(si)
                if not unique_si:
                    continue

                # Tentatively remove g
                for si in block_tsubs[g]:
                    total_count[si] -= 1

                # Find blocks that cover all unique_si
                best_gain = 0
                best_g = -1
                for g_new in range(N_blocks):
                    if in_sol[g_new]:
                        continue
                    # Check if g_new covers all unique_si
                    ok = True
                    for si in unique_si:
                        if si not in block_tsubs_set[g_new]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Count how many previously-uncovered t-subsets it now covers
                    gain = 0
                    for si in block_tsubs[g_new]:
                        if total_count[si] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_g = g_new

                if best_g >= 0:
                    # Apply swap: remove g, add best_g
                    in_sol[g] = False
                    in_sol[best_g] = True
                    for si in block_tsubs[best_g]:
                        total_count[si] += 1
                    selected_indices[i] = best_g
                    improved = True
                else:
                    # Revert
                    for si in block_tsubs[g]:
                        total_count[si] += 1

        return selected_indices

    def simulated_annealing(selected_indices, time_budget=30.0):
        """
        Simulated annealing with fixed-size state.
        State: a list of blocks (fixed size n).
        Energy: number of uncovered t-subsets + penalty * (size - target_size)
        Moves: swap one block for another random block.
        """
        if not selected_indices:
            return selected_indices

        n = len(selected_indices)
        sa_start = time.time()
        sa_deadline = sa_start + time_budget

        # Initial state
        current = list(selected_indices)
        in_sol = np.zeros(N_blocks, dtype=np.bool_)
        for g in current:
            in_sol[g] = True
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Compute initial energy: uncovered count
        uncovered = int(np.sum(total_count == 0))
        current_energy = float(uncovered)
        best_energy = current_energy
        best_state = list(current)

        # Temperature schedule
        T0 = 10.0
        T_end = 0.1
        # We'll do as many iterations as we can
        # Estimate: each iteration is O(k*C(k,t) + C(k,t)) ~ O(126) for update + check
        # Let's target ~50000 iterations in 30s -> ~0.6ms per iteration
        # But we need to check time frequently

        rng = random.Random(42)

        # Precompute: for each block, its t-subsets as a list (already have block_tsubs)
        # For fast energy delta computation, we need to know which t-subsets are uniquely covered

        iterations = 0
        while time.time() < sa_deadline:
            iterations += 1

            # Choose a random block to swap out
            i = rng.randrange(n)
            g_old = current[i]

            # Choose a random block to swap in
            g_new = rng.randrange(N_blocks)
            if g_new == g_old:
                continue

            # Compute energy delta
            # Removing g_old: for each si in block_tsubs[g_old], total_count[si] decreases by 1
            # Adding g_new: for each si in block_tsubs[g_new], total_count[si] increases by 1

            # Delta uncovered:
            # For si in g_old only (not in g_new): if total_count[si] was 1, it becomes 0 -> +1 uncovered
            # For si in g_new only (not in g_old): if total_count[si] was 0, it becomes 1 -> -1 uncovered
            # For si in both: no net change in count, but we need to check if count was 0 before adding g_new...
            # Actually, since g_old is in the solution, total_count[si] >= 1 for si in g_old.
            # After removing g_old, total_count[si] decreases by 1. If it was 1, it becomes 0.
            # Then adding g_new: if si is also in g_new, total_count[si] increases by 1 again.

            # Let's compute delta_uncovered:
            delta_uncovered = 0

            # First, consider removing g_old
            for si in block_tsubs[g_old]:
                if total_count[si] == 1:
                    # This si will become uncovered when g_old is removed, unless g_new also covers it
                    if si not in block_tsubs_set[g_new]:
                        delta_uncovered += 1

            # Then, consider adding g_new
            for si in block_tsubs[g_new]:
                # After removing g_old, the count is total_count[si] - (1 if si in g_old else 0)
                count_after_remove = total_count[si] - (1 if si in block_tsubs_set[g_old] else 0)
                if count_after_remove == 0:
                    # This si was uncovered (after removing g_old), now covered by g_new
                    delta_uncovered -= 1

            delta_energy = float(delta_uncovered)

            # Accept or reject
            if delta_energy <= 0:
                accept = True
            else:
                # Metropolis criterion
                # T decreases linearly with iterations
                # We don't know total iterations, so use time-based cooling
                elapsed = time.time() - sa_start
                progress = min(1.0, elapsed / time_budget)
                T = T0 * (T_end / T0) ** progress
                if T > 0:
                    accept = rng.random() < np.exp(-delta_energy / T)
                else:
                    accept = False

            if accept:
                # Apply the swap
                in_sol[g_old] = False
                in_sol[g_new] = True
                for si in block_tsubs[g_old]:
                    total_count[si] -= 1
                for si in block_tsubs[g_new]:
                    total_count[si] += 1
                current[i] = g_new
                current_energy += delta_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = list(current)
            else:
                # Reject: do nothing
                pass

        # After SA, reduce the best state
        result = reduce_solution_fast(best_state)
        return result

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol_idx = list(sol_idx)
    best_len = len(best_sol_idx)

    # Randomized restarts with SA improvement
    while time.time() < DEADLINE - 15.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol_idx = list(cand_idx)
            # Try local search on the new best
            if time.time() < DEADLINE - 20.0:
                improved = local_search(list(cand_idx))
                improved = reduce_solution_fast(improved)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_sol_idx = list(improved)

    # Final local search pass on the best solution
    if time.time() < DEADLINE - 10.0:
        improved = local_search(list(best_sol_idx))
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    # Simulated annealing polish on the best solution
    if time.time() < DEADLINE - 5.0:
        sa_budget = min(25.0, DEADLINE - time.time() - 5.0)
        if sa_budget > 5.0:
            improved = simulated_annealing(list(best_sol_idx), time_budget=sa_budget)
            if len(improved) < best_len:
                best_len = len(improved)
                best_sol_idx = list(improved)
                # One more local search after SA
                if time.time() < DEADLINE - 2.0:
                    improved2 = local_search(list(best_sol_idx))
                    improved2 = reduce_solution_fast(improved2)
                    if len(improved2) < best_len:
                        best_len = len(improved2)
                        best_sol_idx = list(improved2)

    return [all_blocks[i] for i in best_sol_idx]
