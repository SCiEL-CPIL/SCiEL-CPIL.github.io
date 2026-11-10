# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping. Vectorized."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Non-overlap: for i < j, must not have |dx| < t AND |dy| < t where t = (s_i+s_j)/2 - tol
    dx = np.abs(centers[:, 0][:, None] - centers[:, 0][None, :])
    dy = np.abs(centers[:, 1][:, None] - centers[:, 1][None, :])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    # Only check upper triangle
    iu = np.triu_indices(n, k=1)
    if np.any((dx[iu] < t[iu]) & (dy[iu] < t[iu])):
        return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _max_side_at(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length for square i centered at (cx,cy)
    given the positions and sizes of all other squares. Vectorized.
    """
    # Bound from unit square
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0

    # For each other square j, compute the max s that avoids overlap
    mask = np.arange(len(sides)) != i
    dx = np.abs(cx - centers[mask, 0])
    dy = np.abs(cy - centers[mask, 1])
    sj = sides[mask]
    bounds = np.maximum(2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj)
    if len(bounds) > 0:
        max_s = min(max_s, float(np.min(bounds)))

    if max_s < 0:
        max_s = 0.0
    return max_s


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    """
    n = len(sides)
    for iteration in range(500):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t < 0:
                    continue
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Push apart
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) / 2.0 + 2 * tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        push = (t - abs(dy)) / 2.0 + 2 * tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers


def _grow_all_squares(centers, sides, tol=1e-7, max_rounds=100):
    """
    Iteratively grow all squares to their maximum given current positions.
    Repeat until convergence.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth surrogate for the nonsmooth overlap constraint using log-sum-exp.
    Returns a penalty that is ~0 when no overlap, and large when overlaps exist.
    """
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t < 0:
                continue
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            a = beta * (t - dx)
            b = beta * (t - dy)
            m = max(a, b)
            if m > 700:
                soft_min = m / beta
            else:
                soft_min = (m - math.log(math.exp(a - m) + math.exp(b - m))) / beta
            if soft_min > 0:
                penalty += soft_min ** 2
    return penalty


def _local_search(centers, sides, tol=1e-7, n_iter=200, rng=None, deadline=None):
    """
    Local search: perturb positions, then grow all squares.
    Keep the best result.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Perturb a random subset of squares
        k = int(rng.integers(1, min(n, 5)))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.01, 0.08)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)

        trial_c = _clamp_to_unit(trial_c, best_s)

        if not _is_valid(trial_c, best_s, tol):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol):
                trial_c = trial_c2
            else:
                continue

        # Grow all squares
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol)

        if _is_valid(trial_c, trial_s, tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()

    return best_c, best_s


def _simulated_annealing(centers, sides, tol=1e-7, n_iter=2000, rng=None,
                         t_start=0.5, t_end=0.001, deadline=None):
    """
    Simulated annealing with repulsion potential for temporary overlaps.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)

    curr_c = centers.copy()
    curr_s = sides.copy()

    cooling_rate = (t_end / t_start) ** (1.0 / max(n_iter - 1, 1))

    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break

        T = t_start * (cooling_rate ** it)

        # Perturb: move a random subset of squares
        k = int(rng.integers(1, 4))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = curr_c.copy()
        for i in idx:
            step = rng.uniform(0.02, 0.1)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)

        trial_c = _clamp_to_unit(trial_c, curr_s)

        # Try to grow squares from this position
        trial_s = curr_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=30)

        if _is_valid(trial_c, trial_s, tol=tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()

            curr_c = trial_c.copy()
            curr_s = trial_s.copy()
        else:
            curr_penalty = _smooth_penalty(curr_c, curr_s, tol=tol, beta=100.0)
            trial_penalty = _smooth_penalty(trial_c, trial_s, tol=tol, beta=100.0)

            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, trial_s, tol=tol)
            trial_c2 = _clamp_to_unit(trial_c2, trial_s)
            if _is_valid(trial_c2, trial_s, tol=tol):
                obj = _objective(trial_c2, trial_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_c = trial_c2.copy()
                    best_s = trial_s.copy()
                curr_c = trial_c2.copy()
                curr_s = trial_s.copy()
            else:
                delta = trial_penalty - curr_penalty
                if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                    curr_c = trial_c.copy()
                    curr_s = trial_s.copy()

    return best_c, best_s


def _high_precision_polish(centers, sides, tol=1e-9, max_rounds=500, deadline=None):
    """
    Final high-precision polish: grow squares with tighter tolerance and more iterations.
    Also do a few more local search steps with smaller steps.
    """
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)

    # First, grow with tight tolerance
    trial_s = best_s.copy()
    trial_s = _grow_all_squares(best_c, trial_s, tol=tol, max_rounds=max_rounds)
    if _is_valid(best_c, trial_s, tol=1e-7):
        obj = _objective(best_c, trial_s)
        if obj > best_obj:
            best_obj = obj
            best_s = trial_s.copy()

    # Small-step local search with tight tolerance
    rng = np.random.default_rng(99999)
    for it in range(150):
        if deadline is not None and time.time() > deadline:
            break
        k = int(rng.integers(1, 4))
        idx = rng.choice(10, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.001, 0.01)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)

        trial_c = _clamp_to_unit(trial_c, best_s)

        if not _is_valid(trial_c, best_s, tol=1e-7):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol=1e-7)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol=1e-7):
                trial_c = trial_c2
            else:
                continue

        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=max_rounds)

        if _is_valid(trial_c, trial_s, tol=1e-7):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()

    return best_c, best_s


def _init_symmetric_pattern1():
    """
    4 large squares in corners, 6 smaller filling the cross.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)

    s_large = 0.38
    centers[0] = [s_large / 2, s_large / 2]
    centers[1] = [1 - s_large / 2, s_large / 2]
    centers[2] = [s_large / 2, 1 - s_large / 2]
    centers[3] = [1 - s_large / 2, 1 - s_large / 2]
    sides[0:4] = s_large

    s_small = 0.20
    centers[4] = [0.5, s_large + s_small / 2]
    centers[5] = [0.5, 1 - s_large - s_small / 2]
    centers[6] = [s_large + s_small / 2, 0.5]
    centers[7] = [1 - s_large - s_small / 2, 0.5]
    centers[8] = [0.5, 0.5]
    sides[4:9] = s_small

    centers[9] = [0.5, 0.5 + s_small]
    sides[9] = s_small * 0.8

    return centers, sides


def _init_symmetric_pattern2():
    """
    2x5 grid: 2 rows of 5.
    Side = 1/5 = 0.2
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * 0.2
            centers[idx, 1] = (row + 0.5) * 0.5
            idx += 1
    return centers, sides


def _init_symmetric_pattern3():
    """
    One large (0.5) in bottom-left, 3x3 grid of small (1/6) in the L-shape.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0 / 6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_symmetric_pattern4():
    """
    1 large (0.44) center, 8 around.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.44
    s_small = 0.28
    positions = [
        [0.14, 0.14], [0.86, 0.14], [0.14, 0.86], [0.86, 0.86],
        [0.5, 0.14], [0.5, 0.86], [0.14, 0.5], [0.86, 0.5]
    ]
    for i, pos in enumerate(positions):
        centers[i + 1] = pos
        sides[i + 1] = s_small
    return centers, sides


def _init_symmetric_pattern5():
    """
    5 large in a cross, 5 small in corners.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.25
    centers[1] = [0.5, 0.25]
    centers[2] = [0.5, 0.75]
    centers[3] = [0.25, 0.5]
    centers[4] = [0.75, 0.5]
    sides[1:5] = 0.25
    s_small = 0.15
    centers[5] = [0.125, 0.125]
    centers[6] = [0.875, 0.125]
    centers[7] = [0.125, 0.875]
    centers[8] = [0.875, 0.875]
    centers[9] = [0.5, 0.05]
    sides[5:10] = s_small
    return centers, sides


def _init_random(rng, n=10):
    """
    Generate a random initial configuration of 10 small squares.
    """
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = rng.uniform(0.05, 0.25)
        sides[i] = s
        centers[i, 0] = rng.uniform(s / 2, 1 - s / 2)
        centers[i, 1] = rng.uniform(s / 2, 1 - s / 2)
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 570.0

    inits = [
        _init_symmetric_pattern1(),
        _init_symmetric_pattern2(),
        _init_symmetric_pattern3(),
        _init_symmetric_pattern4(),
        _init_symmetric_pattern5(),
    ]

    # Add a few random inits for diversity
    rng_init = np.random.default_rng(777)
    for _ in range(3):
        inits.append(_init_random(rng_init))

    best_centers = None
    best_sides = None
    best_obj = -1.0

    rng_global = np.random.default_rng(12345)

    for init_idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break

        centers = centers.copy()
        sides = sides.copy()

        # Ensure validity
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides, tol=1e-7):
            centers = _resolve_overlap(centers, sides, tol=1e-7)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides, tol=1e-7):
                continue

        # Grow all squares
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        if not _is_valid(centers, sides, tol=1e-7):
            continue

        obj = _objective(centers, sides)
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            best_sides = sides.copy()

        # Local search from this point
        if time.time() < deadline - 10:
            rng = np.random.default_rng(init_idx * 7919 + 1)
            opt_c, opt_s = _local_search(centers, sides, tol=1e-7, n_iter=200, rng=rng, deadline=deadline - 10)
            if _is_valid(opt_c, opt_s, tol=1e-7):
                obj2 = _objective(opt_c, opt_s)
                if obj2 > best_obj:
                    best_obj = obj2
                    best_centers = opt_c.copy()
                    best_sides = opt_s.copy()

        # Simulated annealing from this point (if time permits)
        if time.time() < deadline - 30:
            rng_sa = np.random.default_rng(init_idx * 104729 + 7)
            sa_c, sa_s = _simulated_annealing(
                centers, sides, tol=1e-7,
                n_iter=800, rng=rng_sa,
                t_start=0.4, t_end=0.003,
                deadline=deadline - 5
            )
            if _is_valid(sa_c, sa_s, tol=1e-7):
                obj3 = _objective(sa_c, sa_s)
                if obj3 > best_obj:
                    best_obj = obj3
                    best_centers = sa_c.copy()
                    best_sides = sa_s.copy()

    # If nothing valid, fallback
    if best_centers is None:
        centers, sides = _init_symmetric_pattern2()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        best_centers = centers
        best_sides = sides
    else:
        # High-precision final polish with tighter tolerance
        if time.time() < deadline - 5:
            polished_c, polished_s = _high_precision_polish(best_centers, best_sides, tol=1e-9, max_rounds=500, deadline=deadline - 2)
            if _is_valid(polished_c, polished_s, tol=1e-7):
                obj_polished = _objective(polished_c, polished_s)
                if obj_polished > best_obj:
                    best_centers = polished_c
                    best_sides = polished_s
                    best_obj = obj_polished

    return best_centers, best_sides


# EVOLVE-BLOCK-END
