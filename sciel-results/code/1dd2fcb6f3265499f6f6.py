# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    No 5 on a sphere in the 8x8x8 grid.
    """
    n = 8
    deadline = time.monotonic() + 600.0
    safety_margin = 15.0

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)
    N = len(grid)

    # Precompute lifted coordinates: for point i, lifted = (x, y, z, x^2+y^2+z^2)
    # For the determinant test: given candidate p and existing points,
    # we need det of 4x4 matrix with rows (q_i - p, |q_i|^2 - |p|^2)
    # which equals det of [q_i - p; |q_i|^2 - |p|^2] for 4 points.

    def can_add_fast(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Check if p can be added to arr without creating 5 co-spherical points."""
        m = len(arr)
        if m < 4:
            return True
        # Build lifted vectors relative to p
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        # Check all C(m,4) combinations
        # For small m this is fine; for larger m we use early termination
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    def can_add_batched(arr: np.ndarray, arr_sq: np.ndarray, p: np.ndarray, p_sq: float) -> bool:
        """Vectorized version for larger m."""
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - p_sq

        combos = list(itertools.combinations(range(m), 4))
        if len(combos) <= 5000:
            # Batch determinant
            sub = w[np.array(combos)]  # (C, 4, 4)
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 0.5)
        else:
            # Fallback to sequential for very large m
            for combo in combos:
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True

    def greedy_pass(order: np.ndarray, deadline: float, use_batched: bool = True) -> tuple:
        """Run a greedy pass over the given order of points."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        skipped = []

        for p in order:
            if time.monotonic() + safety_margin >= deadline:
                break
            p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
            if use_batched:
                ok = can_add_batched(arr, arr_sq, p, p_sq)
            else:
                ok = can_add_fast(arr, arr_sq, p, p_sq)
            if ok:
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([p_sq])
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
            else:
                skipped.append(p)

        return arr, arr_sq, skipped

    def local_improve(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float) -> tuple:
        """Try to insert previously skipped points."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        improved = True
        while improved and (time.monotonic() + safety_margin < deadline):
            improved = False
            for i in range(len(skipped)):
                if time.monotonic() + safety_margin >= deadline:
                    break
                p = skipped[i]
                p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                if can_add_batched(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.pop(i)
                    improved = True
                    # Restart scan from beginning after insertion
                    break
        return arr, arr_sq, skipped

    def local_improve_multi(arr: np.ndarray, arr_sq: np.ndarray, skipped: list, deadline: float) -> tuple:
        """More aggressive local improvement: multiple passes with shuffled skipped list."""
        arr = arr.copy()
        arr_sq = arr_sq.copy()
        rng_local = np.random.default_rng(42)
        
        for _pass in range(20):
            if time.monotonic() + safety_margin >= deadline:
                break
            if not skipped:
                break
            # Shuffle skipped
            order_skipped = list(skipped)
            rng_local.shuffle(order_skipped)
            improved_this_pass = False
            for p in order_skipped:
                if time.monotonic() + safety_margin >= deadline:
                    break
                p_sq = grid_sq[np.where((grid == p).all(axis=1))[0][0]]
                if can_add_batched(arr, arr_sq, p, p_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, p_sq)
                    skipped.remove(p)
                    improved_this_pass = True
            if not improved_this_pass:
                break
        return arr, arr_sq, skipped

    # Generate diverse orderings
    def make_random_order(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return grid[rng.permutation(N)]

    def make_norm_sorted_order(reverse: bool = False) -> np.ndarray:
        """Sort by squared norm (distance from origin)."""
        idx = np.argsort(grid_sq)
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    def make_corner_biased_order(seed: int) -> np.ndarray:
        """Bias toward corners and edges."""
        rng = np.random.default_rng(seed)
        # Score: prefer points with more coordinates at 0 or 7
        corner_score = np.zeros(N)
        for i, p in enumerate(grid):
            for c in p:
                if c == 0 or c == 7:
                    corner_score[i] += 1
        # Mix corner score with random
        noise = rng.random(N) * 0.5
        scores = corner_score + noise
        idx = np.argsort(scores)
        return grid[idx]

    def make_axis_sorted_order(axis: int, reverse: bool = False) -> np.ndarray:
        """Sort by one axis coordinate."""
        idx = np.argsort(grid[:, axis])
        if reverse:
            idx = idx[::-1]
        return grid[idx]

    best = None
    best_sq = None

    # Phase 1: Quick greedy passes with diverse orderings
    orderings = []
    # Random orderings
    for seed in range(10):
        orderings.append(('random', make_random_order(seed)))
    # Norm sorted
    orderings.append(('norm_asc', make_norm_sorted_order(False)))
    orderings.append(('norm_desc', make_norm_sorted_order(True)))
    # Corner biased
    for seed in range(5):
        orderings.append(('corner', make_corner_biased_order(seed)))
    # Axis sorted
    for axis in range(3):
        orderings.append((f'axis{axis}_asc', make_axis_sorted_order(axis, False)))
        orderings.append((f'axis{axis}_desc', make_axis_sorted_order(axis, True)))

    for name, order in orderings:
        if time.monotonic() + safety_margin >= deadline:
            break
        arr, arr_sq, skipped = greedy_pass(order, deadline, use_batched=True)
        # Light local improvement
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline)

        if best is None or len(arr) > len(best):
            best = arr
            best_sq = arr_sq

    # Phase 2: More intensive local improvement on the best solution
    if best is not None and time.monotonic() + safety_margin < deadline:
        # Collect all points not in best
        best_set = set(map(tuple, best))
        all_points = [p for p in grid if tuple(p) not in best_set]
        
        # Try to add more points with multi-pass local improvement
        best, best_sq, _ = local_improve_multi(best, best_sq, all_points, deadline)

    # Phase 3: Try additional random restarts if time remains
    extra_restarts = 30
    for i in range(extra_restarts):
        if time.monotonic() + safety_margin >= deadline:
            break
        order = make_random_order(100 + i)
        arr, arr_sq, skipped = greedy_pass(order, deadline, use_batched=True)
        arr, arr_sq, skipped = local_improve(arr, arr_sq, skipped, deadline)

        if best is None or len(arr) > len(best):
            best = arr
            best_sq = arr_sq

    if best is None:
        best = np.empty((0, 3), dtype=np.int64)

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
