# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_f = grid.astype(np.float64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    # Pre-compute combination indices for m=4 to m=80
    combos_cache = {}
    for m in range(4, 81):
        idx = np.arange(m)
        combos = list(itertools.combinations(idx, 4))
        if combos:
            combos_cache[m] = np.array(combos, dtype=np.int32)
        else:
            combos_cache[m] = np.empty((0, 4), dtype=np.int32)

    def can_add(arr_indices, p_idx):
        """
        arr_indices: list of indices of points already in the set.
        p_idx: index of candidate point to add.
        Returns True if adding p_idx keeps the set valid.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        p = grid_f[p_idx]
        p_sq = grid_sq[p_idx]

        # Build matrix: rows[i] = (arr[i] - p, |arr[i]|^2 - |p|^2)
        arr_pts = grid_f[arr_indices]  # (m, 3)
        diffs = arr_pts - p  # (m, 3)
        sq_diffs = grid_sq[arr_indices] - p_sq  # (m,)
        rows = np.column_stack([diffs, sq_diffs])  # (m, 4)

        if m not in combos_cache:
            # Fallback for very large m
            idx = np.arange(m)
            combos = list(itertools.combinations(idx, 4))
            if not combos:
                return True
            combo_arr = np.array(combos, dtype=np.int32)
        else:
            combo_arr = combos_cache[m]
            
        if combo_arr.shape[0] == 0:
            return True

        # Process in batches to limit memory usage
        batch_size = 5000
        for start in range(0, combo_arr.shape[0], batch_size):
            end = min(start + batch_size, combo_arr.shape[0])
            sub = rows[combo_arr[start:end]]  # (batch, 4, 4)
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        return True

    def can_add_fast(arr_indices, p_idx, m_limit=30):
        """
        Faster check for smaller sets. For larger sets, use can_add.
        """
        m = len(arr_indices)
        if m < 4:
            return True
        if m <= m_limit:
            return can_add(arr_indices, p_idx)
        # For larger m, do a subset check first (heuristic)
        # Check a random subset of 4-tuples
        if m > 40:
            # Sample some 4-tuples
            n_samples = min(2000, m * (m-1) * (m-2) * (m-3) // 24)
            sampled = rng_integers(4, size=(n_samples, 4), low=0, high=m)
            p = grid_f[p_idx]
            p_sq = grid_sq[p_idx]
            arr_pts = grid_f[arr_indices]
            diffs = arr_pts - p
            sq_diffs = grid_sq[arr_indices] - p_sq
            rows = np.column_stack([diffs, sq_diffs])
            sub = rows[sampled]
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
            # If sample passes, do full check
            return can_add(arr_indices, p_idx)
        return can_add(arr_indices, p_idx)

    # Pre-generate random integers for sampling
    _rng_local = np.random.default_rng(12345)
    def rng_integers(size, low, high):
        return _rng_local.integers(low, high, size=size)

    def greedy_fill(order):
        """Greedy fill: add points in given order if valid."""
        arr_indices = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            if can_add(arr_indices, idx):
                arr_indices.append(idx)
        return arr_indices

    def local_search(arr_indices, max_rounds=30, candidate_limit=100):
        """
        Swap-based local search: try to improve by removing one point and adding another.
        Limits candidates tried per removal for speed.
        """
        if len(arr_indices) < 4:
            return arr_indices

        current = list(arr_indices)
        current_set = set(current)
        improved = True
        rounds = 0

        while improved and rounds < max_rounds and time.time() < deadline - 30:
            improved = False
            rounds += 1
            for i in range(len(current)):
                if time.time() > deadline - 30:
                    break
                # Remove point i
                remaining = [x for x in current if x != current[i]]
                remaining_set = set(remaining)

                # Try adding candidates - limit to candidate_limit for speed
                candidates = [j for j in range(N) if j not in remaining_set]
                if len(candidates) > candidate_limit:
                    rng_local = np.random.default_rng(rounds * 100 + i)
                    candidates = list(rng_local.choice(candidates, size=candidate_limit, replace=False))

                for j in candidates:
                    if can_add(remaining, j):
                        new_list = remaining + [j]
                        new_set = set(new_list)
                        if new_set != current_set:
                            current = new_list
                            current_set = new_set
                            improved = True
                            break  # restart scan after a swap
                # Continue scanning other removals in this round
        return current

    # --- Main search loop ---
    rng = np.random.default_rng(42)
    best_set = []
    best_size = 0

    num_restarts = 100

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Vary ordering strategy
        r = restart % 4
        if r == 0:
            order = rng.permutation(N)
        elif r == 1:
            # Bias toward corners
            dist = np.sum((grid_f - 3.5) ** 2, axis=1)
            weights = dist + 1e-9
            weights /= weights.sum()
            order = rng.choice(N, size=N, replace=False, p=weights)
        elif r == 2:
            # Bias toward center
            dist = np.sum((grid_f - 3.5) ** 2, axis=1)
            weights = (dist.max() + 1.0 - dist) + 1e-9
            weights /= weights.sum()
            order = rng.choice(N, size=N, replace=False, p=weights)
        else:
            # Random
            order = rng.permutation(N)

        # Greedy fill
        current = greedy_fill(order)

        # Local search improvement
        if len(current) >= 4:
            current = local_search(current, max_rounds=20, candidate_limit=80)

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

        # Simulated Annealing phase on current best
        if len(current) >= 4 and time.time() < deadline - 60:
            sa_set = list(current)
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            sa_steps = 0
            sa_max_steps = 5000

            while sa_steps < sa_max_steps and time.time() < deadline - 30:
                sa_steps += 1
                move_type = rng.random()

                if move_type < 0.4:
                    # Try add
                    sa_set_set = set(sa_set)
                    candidates = [i for i in range(N) if i not in sa_set_set]
                    if candidates:
                        move_idx = int(rng.choice(candidates))
                        if can_add(sa_set, move_idx):
                            sa_set.append(move_idx)
                            if len(sa_set) > best_size:
                                best_size = len(sa_set)
                                best_set = list(sa_set)
                elif move_type < 0.7:
                    # Try remove
                    if sa_set:
                        move_idx = int(rng.choice(len(sa_set)))
                        # Accept removal with some probability (to escape local optima)
                        if rng.random() < np.exp(-0.5 / T):
                            sa_set.pop(move_idx)
                else:
                    # Try swap
                    if sa_set and len(sa_set) < N:
                        out_pos = int(rng.choice(len(sa_set)))
                        out_idx = sa_set[out_pos]
                        sa_set_set = set(sa_set)
                        candidates = [i for i in range(N) if i not in sa_set_set]
                        if candidates:
                            in_idx = int(rng.choice(candidates))
                            # Check: remove out_idx, add in_idx
                            remaining = [x for x in sa_set if x != out_idx]
                            if can_add(remaining, in_idx):
                                sa_set[out_pos] = in_idx

                T *= cooling
                if T < min_T:
                    T = min_T

    # Convert best set to array
    if best_size > 0:
        result = np.array([grid[i] for i in best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
