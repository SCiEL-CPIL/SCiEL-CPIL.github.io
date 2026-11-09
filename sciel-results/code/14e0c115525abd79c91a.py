# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    Maximize the number of points from {0,...,7}^3 such that no 5 are co-spherical.
    """
    n = 8
    deadline = time.monotonic() + 620.0
    safety_margin = 5.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.int64)
    N = len(grid)

    def can_add(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: int) -> bool:
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.int64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq
        combos = list(itertools.combinations(range(m), 4))
        batch_size = 3000
        for start in range(0, len(combos), batch_size):
            end = min(start + batch_size, len(combos))
            batch = combos[start:end]
            idx = np.array(batch, dtype=np.int64)
            sub = w[idx].astype(np.float64)
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        return True

    def greedy_pass(order: np.ndarray, deadline: float) -> tuple:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        skipped = []
        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = int(grid_sq[list(itertools.product(range(n),range(n),range(n))).index(tuple(p))] if False else 0)
            # Use direct lookup
            p_sq = int((p ** 2).sum())
            if can_add(arr, arr_sq, p, p_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq], dtype=np.int64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append(p)
        return arr, arr_sq, skipped

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, candidates: list, deadline: float, max_passes: int = 100) -> tuple:
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        remaining = list(candidates)
        rng_local = np.random.default_rng(42)
        for _pass in range(max_passes):
            if time.monotonic() + safety_margin >= deadline:
                break
            if not remaining:
                break
            rng_local.shuffle(remaining)
            improved = False
            for i, p in enumerate(remaining):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p_sq = int((p ** 2).sum())
                if can_add(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    remaining.pop(i)
                    improved = True
            if not improved:
                break
        return arr, arr_sq, remaining

    def make_random_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return grid[rng.permutation(N)]

    def make_norm_order(reverse: bool = False) -> np.ndarray:
        idx = np.argsort(grid_sq)
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_corner_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        corner_score = np.zeros(N)
        for i, p in enumerate(grid):
            for c in p:
                if c == 0 or c == 7:
                    corner_score[i] += 1
        noise = rng.random(N) * 0.5
        idx = np.argsort(corner_score + noise)
        return grid[idx]

    def make_axis_order(axis: int, reverse: bool = False) -> np.ndarray:
        idx = np.argsort(grid[:, axis])
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_diag_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        diag = grid.sum(axis=1).astype(float) + rng.random(N) * 0.1
        idx = np.argsort(diag)
        return grid[idx]

    best = None
    best_sq = None

    # Phase 1: Diverse greedy passes
    orderings = []
    for s in range(10):
        orderings.append(make_random_order(s))
    orderings.append(make_norm_order(False))
    orderings.append(make_norm_order(True))
    for s in range(5):
        orderings.append(make_corner_order(s))
    for ax in range(3):
        orderings.append(make_axis_order(ax, False))
        orderings.append(make_axis_order(ax, True))
    for s in range(5):
        orderings.append(make_diag_order(s))

    for order in orderings:
        if time.monotonic() + safety_margin >= deadline:
            break
        arr, arr_sq, skipped = greedy_pass(order, deadline)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline, max_passes=10)
        if best is None or len(arr) > len(best):
            best = arr.copy()
            best_sq = arr_sq.copy()

    # Phase 2: Intensive polish on best
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        all_pts = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = local_improve(best, best_sq, all_pts, deadline, max_passes=200)

    # Phase 3: More random restarts
    for i in range(100):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = make_random_order(200 + i)
        arr, arr_sq, skipped = greedy_pass(order, deadline)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline, max_passes=15)
        if best is None or len(arr) > len(best):
            best = arr.copy()
            best_sq = arr_sq.copy()

    # Phase 4: Final polish
    if best is not None and time.monotonic() + safety_margin < deadline:
        best_set = set(map(tuple, best))
        all_pts = [p for p in grid if tuple(p) not in best_set]
        best, best_sq, _ = local_improve(best, best_sq, all_pts, deadline, max_passes=500)

    if best is None:
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
