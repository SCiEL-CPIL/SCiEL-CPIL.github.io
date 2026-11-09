# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical.
    Uses LP relaxation + greedy + local search.
    """
    n = 8
    deadline = time.time() + 580  # 20s safety margin

    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point

    def is_cospherical(i, j, k, l, m):
        """Check if 5 points are co-spherical using the determinant test."""
        p = grid[i]
        pi_sq = sq[i]
        pts = np.array([grid[j], grid[k], grid[l], grid[m]], dtype=np.float64)
        sqs = np.array([sq[j], sq[k], sq[l], sq[m]], dtype=np.float64)
        w = np.empty((4, 4), dtype=np.float64)
        w[:, 0:3] = pts - p
        w[:, 3] = sqs - pi_sq
        return abs(np.linalg.det(w)) < 0.5

    # Precompute all cospherical 5-tuples
    print("Computing cospherical 5-tuples...")
    cospherical_tuples = []
    count = 0
    for combo in itertools.combinations(range(N), 5):
        i, j, k, l, m = combo
        if is_cospherical(i, j, k, l, m):
            cospherical_tuples.append(combo)
            count += 1
        if count % 10000 == 0 and count > 0:
            print(f"  Found {count} cospherical tuples so far...")
    print(f"Total cospherical 5-tuples: {count}")

    # Build conflict graph: edge between two points if they appear together in a cospherical 5-tuple
    # Actually, we need: for each 5-tuple, all pairs within it conflict
    conflict_pairs = set()
    for combo in cospherical_tuples:
        for a in range(5):
            for b in range(a + 1, 5):
                conflict_pairs.add((combo[a], combo[b]))
    print(f"Total conflict pairs: {len(conflict_pairs)}")

    # Build adjacency for each point
    adj = [set() for _ in range(N)]
    for (a, b) in conflict_pairs:
        adj[a].add(b)
        adj[b].add(a)

    def can_add_to_set(arr_indices: np.ndarray, new_idx: int) -> bool:
        """Check if adding new_idx to the set keeps it valid (no 5 co-spherical)."""
        m = len(arr_indices)
        if m < 4:
            return True
        p = grid[new_idx]
        pi_sq = sq[new_idx]
        others = arr_indices
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = grid[others] - p
        w[:, 3] = sq[others] - pi_sq
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def greedy_build(order: np.ndarray, start_arr: np.ndarray = None) -> np.ndarray:
        """Greedy: add points in given order if feasible."""
        if start_arr is None:
            arr = np.empty(0, dtype=np.int64)
        else:
            arr = start_arr.copy()
        arr_set = set(arr.tolist())
        for idx in order:
            if idx in arr_set:
                continue
            if can_add_to_set(arr, idx):
                arr = np.append(arr, idx)
                arr_set.add(idx)
        return arr

    def local_search(arr: np.ndarray, max_swaps: int = 200) -> np.ndarray:
        """Try swap moves: remove one, add another, keep if valid."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)

        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without = np.delete(current, i)
                for j in range(len(comp)):
                    if time.time() > deadline:
                        break
                    add_idx = comp[j]
                    if can_add_to_set(without, add_idx):
                        new_arr = np.concatenate([without, [add_idx]])
                        current = new_arr
                        current_set = set(current.tolist())
                        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                        improved = True
                        swaps_done += 1
                        break
                if improved:
                    break
        return current

    def local_search_add(arr: np.ndarray) -> np.ndarray:
        """Try to add more points after local search."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)

        for idx in comp:
            if time.time() > deadline:
                break
            if can_add_to_set(current, idx):
                current = np.append(current, idx)
                current_set.add(idx)
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        return current

    # Phase 0: LP relaxation to get a good upper bound and initial solution
    print("Solving LP relaxation...")
    lp_start = time.time()
    # Objective: maximize sum x_i
    c = -np.ones(N)  # minimize -sum x_i
    # Constraints: for each cospherical 5-tuple, sum of its 5 variables <= 4
    A_ub_list = []
    b_ub_list = []
    for combo in cospherical_tuples:
        row = np.zeros(N)
        for idx in combo:
            row[idx] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(4.0)

    if len(A_ub_list) > 0:
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
    else:
        A_ub = np.zeros((0, N))
        b_ub = np.zeros(0)

    bounds = [(0, 1)] * N
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        lp_time = time.time() - lp_start
        print(f"LP solved in {lp_time:.1f}s")
        if result.success:
            lp_obj = -result.fun
            print(f"LP objective (upper bound): {lp_obj:.2f}")
            # Use LP solution to guide greedy: sort by LP values
            lp_vals = result.x
            order_lp = np.argsort(-lp_vals)
        else:
            print("LP failed, using random order")
            order_lp = np.random.permutation(N)
    except Exception as e:
        print(f"LP error: {e}, using random order")
        order_lp = np.random.permutation(N)

    rng = np.random.RandomState(42)
    best = None
    best_len = 0

    # Phase 1: Greedy with LP-guided order
    print("Phase 1: LP-guided greedy...")
    arr = greedy_build(order_lp)
    print(f"LP-guided greedy found {len(arr)} points")
    if len(arr) > best_len:
        best = arr
        best_len = len(arr)

    # Phase 2: Multiple random greedy restarts
    print("Phase 2: Random greedy restarts...")
    while time.time() < deadline - 120:
        order = rng.permutation(N)
        arr = greedy_build(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)

    # Phase 3: Local search on best
    if best is not None and time.time() < deadline:
        print("Phase 3: Local search...")
        best = local_search(best)
        best = local_search_add(best)
        best_len = len(best)
        print(f"After local search: {best_len} points")

    # Phase 4: More restarts with local search if time permits
    while time.time() < deadline - 60:
        order = rng.permutation(N)
        arr = greedy_build(order)
        arr = local_search(arr, max_swaps=50)
        arr = local_search_add(arr)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
            print(f"Improved to {best_len} points")

    if best is None:
        best = np.array([0, 1, 2, 3], dtype=np.int64)

    print(f"Final result: {len(best)} points")
    return grid[best].astype(np.int64)


# EVOLVE-BLOCK-END
