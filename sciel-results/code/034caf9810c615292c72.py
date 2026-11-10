# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin before 600s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_float = grid.astype(np.float64)
    grid_sq = (grid_float ** 2).sum(axis=1).astype(np.float64)

    def can_add(arr_indices, p_idx):
        """
        Check if p_idx can be added to arr_indices without violating the constraint.
        arr_indices: list of indices of points already in the set.
        p_idx: index of candidate point to add.
        Returns True if adding p_idx keeps the set valid.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        p = grid_float[p_idx]
        p_sq = grid_sq[p_idx]

        arr_pts = grid_float[np.array(arr_indices)]
        diffs = arr_pts - p
        sq_diffs = grid_sq[np.array(arr_indices)] - p_sq
        rows = np.column_stack([diffs, sq_diffs])

        # Use itertools for combinations to avoid precomputation memory overhead for large m
        for combo in itertools.combinations(range(m), 4):
            sub = rows[list(combo)]
            det_val = np.linalg.det(sub)
            if abs(det_val) < 0.5:
                return False
        return True

    def greedy_fill(order):
        """Greedy fill: add points in given order if valid."""
        arr_indices = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            if can_add(arr_indices, idx):
                arr_indices.append(idx)
        return arr_indices

    def local_search(arr_indices, max_rounds=50):
        """
        Swap-based local search: try to improve by removing one point and adding another.
        Returns improved list.
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
                
                # Candidates are all points not in remaining_set
                candidates = [j for j in range(N) if j not in remaining_set]
                rng.shuffle(candidates)
                
                # Limit number of candidates to check to speed up
                for j in candidates[:50]:
                    if can_add(remaining, j):
                        new_list = remaining + [j]
                        new_set = set(new_list)
                        if new_set != current_set:
                            current = new_list
                            current_set = new_set
                            improved = True
                            break  # restart scan after a swap
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
            dist = np.sum((grid_float - 3.5) ** 2, axis=1)
            weights = dist + 1e-9
            weights /= weights.sum()
            order = rng.choice(N, size=N, replace=False, p=weights)
        elif r == 2:
            # Bias toward center
            dist = np.sum((grid_float - 3.5) ** 2, axis=1)
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
            current = local_search(current, max_rounds=20)

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
                    candidates = [i for i in range(N) if i not in sa_set]
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
                        candidates = [i for i in range(N) if i not in sa_set]
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
