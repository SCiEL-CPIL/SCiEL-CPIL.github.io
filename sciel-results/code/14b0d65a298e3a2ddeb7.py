# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1.0 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers


def _grow_square_max(centers, sides, i, tol=1e-7):
    """Find the maximum side length for square i given all other positions/sizes (analytic)."""
    cx, cy = centers[i]
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0
    n = len(sides)
    for j in range(n):
        if j == i:
            continue
        dx = abs(cx - centers[j, 0])
        dy = abs(cy - centers[j, 1])
        c1 = 2.0 * (dx + tol) - sides[j]
        c2 = 2.0 * (dy + tol) - sides[j]
        c = max(c1, c2)
        if c < max_s:
            max_s = c
    if max_s < sides[i]:
        return sides[i]
    return max_s


def _grow_all_to_convergence(centers, sides, tol=1e-7, max_iters=200):
    """Iteratively grow all squares until convergence."""
    n = len(sides)
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _resolve_overlap(centers, sides, tol=1e-7):
    """Attempt to resolve overlaps by pushing squares apart."""
    n = len(sides)
    for iteration in range(300):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) / 2.0 + tol * 2
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        push = (t - abs(dy)) / 2.0 + tol * 2
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


def _try_push_to_corner(centers, sides, i, tol=1e-7):
    """Try to push square i toward corners and edges."""
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    half = sides[i] / 2.0
    targets = [
        (half, half),
        (1.0 - half, half),
        (half, 1.0 - half),
        (1.0 - half, 1.0 - half),
        (0.5, half),
        (0.5, 1.0 - half),
        (half, 0.5),
        (1.0 - half, 0.5),
    ]
    for tx, ty in targets:
        test_c = centers.copy()
        test_c[i, 0] = tx
        test_c[i, 1] = ty
        if not _is_valid(test_c, sides, tol):
            continue
        trial_s = sides.copy()
        for k in range(n):
            new_s = _grow_square_max(test_c, trial_s, k, tol)
            if new_s > trial_s[k] + 1e-9:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    return best_c, best_s


def _local_optimize(centers, sides, tol=1e-7, max_rounds=80):
    """Local optimization with growth, repositioning, and perturbations."""
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    for rnd in range(max_rounds):
        new_sides = sides.copy()
        new_sides = _grow_all_to_convergence(centers, new_sides, tol)
        if _is_valid(centers, new_sides, tol):
            sides = new_sides
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = sides.copy()
                best_centers = centers.copy()
        for i in range(n):
            new_c, new_s = _try_push_to_corner(centers, sides, i, tol)
            if _is_valid(new_c, new_s, tol):
                obj = _objective(new_c, new_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_centers = new_c.copy()
                    best_sides = new_s.copy()
                    centers = new_c.copy()
                    sides = new_s.copy()
        rng = np.random.default_rng(seed=rnd * 1000 + 42)
        for i in range(n):
            for _ in range(8):
                dx = rng.uniform(-0.03, 0.03)
                dy = rng.uniform(-0.03, 0.03)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i] / 2, 1.0 - sides[i] / 2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i] / 2, 1.0 - sides[i] / 2)
                if _is_valid(new_c, sides, tol):
                    trial_s = sides.copy()
                    trial_s = _grow_all_to_convergence(new_c, trial_s, tol)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-9:
                            best_obj = obj
                            best_centers = new_c.copy()
                            best_sides = trial_s.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
        if rnd > 10:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-9:
                    can_improve = True
                    break
            if not can_improve:
                break
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 2x5 grid packing."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    side = 0.2
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * side
            centers[idx, 1] = (row + 0.5) * side
            sides[idx] = side
            idx += 1
    return centers, sides


def _init_heterogeneous_1():
    """One large (0.5) + 9 small (1/6) in L-shape."""
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


def _init_heterogeneous_2():
    """Two large (0.4) + 8 small (0.25)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    side_s = 0.25
    idx = 2
    for row in range(2):
        for col in range(4):
            centers[idx, 0] = (col + 0.5) * side_s
            centers[idx, 1] = 0.4 + (row + 0.5) * side_s
            sides[idx] = side_s
            idx += 1
    return centers, sides


def _init_heterogeneous_3():
    """One large (0.5) + 3 medium (0.25) + 6 small (0.125)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    centers[3] = [0.125, 0.875]
    sides[3] = 0.25
    side_s = 0.125
    idx = 4
    positions = [
        [0.625, 0.375], [0.875, 0.375],
        [0.375, 0.625], [0.375, 0.875],
        [0.625, 0.875], [0.875, 0.875],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """Two large (0.45) in opposite corners + 8 small (0.15)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.775, 0.775]
    sides[1] = 0.45
    side_s = 0.15
    idx = 2
    positions = [
        [0.225, 0.575], [0.225, 0.875],
        [0.525, 0.225], [0.825, 0.225],
        [0.525, 0.525], [0.05, 0.575], [0.05, 0.875], [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """Three large (0.4) in three corners + 7 small (0.2)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    centers[2] = [0.2, 0.8]
    sides[2] = 0.4
    side_s = 0.2
    idx = 3
    positions = [
        [0.5, 0.5], [0.8, 0.5],
        [0.5, 0.8], [0.8, 0.8],
        [0.65, 0.65], [0.65, 0.35], [0.35, 0.65],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """One large (0.5) + 2 medium (0.3) + 7 small, asymmetric."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.85, 0.15]
    sides[1] = 0.3
    centers[2] = [0.15, 0.85]
    sides[2] = 0.3
    side_s = 0.15
    idx = 3
    positions = [
        [0.55, 0.15], [0.55, 0.55], [0.85, 0.55],
        [0.15, 0.55], [0.55, 0.85], [0.85, 0.85], [0.4, 0.4],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """One very large (0.55) + 9 small, asymmetric."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.275, 0.275]
    sides[0] = 0.55
    side_s = 0.145
    idx = 1
    positions = [
        [0.725, 0.145], [0.725, 0.435], [0.725, 0.725],
        [0.145, 0.725], [0.435, 0.725], [0.725, 0.875],
        [0.145, 0.55], [0.435, 0.55], [0.875, 0.275],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_8():
    """Four medium (0.3) in corners + 6 small (0.2)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.15, 0.15]
    sides[0] = 0.3
    centers[1] = [0.85, 0.15]
    sides[1] = 0.3
    centers[2] = [0.15, 0.85]
    sides[2] = 0.3
    centers[3] = [0.85, 0.85]
    sides[3] = 0.3
    side_s = 0.2
    idx = 4
    positions = [
        [0.5, 0.15], [0.5, 0.5], [0.5, 0.85],
        [0.15, 0.5], [0.85, 0.5], [0.5, 0.5],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_9():
    """One large (0.5) + 4 medium (0.2) + 5 small (0.15)."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.1]
    sides[1] = 0.2
    centers[2] = [0.75, 0.7]
    sides[2] = 0.2
    centers[3] = [0.1, 0.75]
    sides[3] = 0.2
    centers[4] = [0.7, 0.4]
    sides[4] = 0.2
    side_s = 0.15
    idx = 5
    positions = [
        [0.4, 0.6], [0.6, 0.6], [0.4, 0.85], [0.6, 0.85], [0.85, 0.55],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_10():
    """Two large (0.45) adjacent + 8 small, different arrangement."""
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.225, 0.225]
    sides[0] = 0.45
    centers[1] = [0.225, 0.775]
    sides[1] = 0.45
    side_s = 0.18
    idx = 2
    positions = [
        [0.79, 0.18], [0.79, 0.42], [0.79, 0.66],
        [0.55, 0.18], [0.55, 0.42], [0.55, 0.66],
        [0.9, 0.3], [0.9, 0.55],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_random(seed, n_large=2, large_min=0.3, large_max=0.5, small_min=0.1, small_max=0.2):
    """Generate a random initial packing."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    for i in range(10):
        if i < n_large:
            s = rng.uniform(large_min, large_max)
        else:
            s = rng.uniform(small_min, small_max)
        sides[i] = s
        centers[i, 0] = rng.uniform(s / 2, 1.0 - s / 2)
        centers[i, 1] = rng.uniform(s / 2, 1.0 - s / 2)
    return centers, sides


def _simulated_annealing(centers, sides, tol=1e-7, start_temp=0.05, end_temp=0.001, n_iter=3000, seed=42):
    """Simulated annealing to find better packings."""
    rng = np.random.default_rng(seed)
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    cur_c = centers.copy()
    cur_s = sides.copy()
    cur_obj = _objective(cur_c, cur_s)
    for it in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / n_iter)
        i = rng.integers(0, n)
        new_c = cur_c.copy()
        dx = rng.uniform(-0.04, 0.04)
        dy = rng.uniform(-0.04, 0.04)
        new_c[i, 0] = np.clip(cur_c[i, 0] + dx, cur_s[i] / 2, 1.0 - cur_s[i] / 2)
        new_c[i, 1] = np.clip(cur_c[i, 1] + dy, cur_s[i] / 2, 1.0 - cur_s[i] / 2)
        if _is_valid(new_c, cur_s, tol):
            trial_s = cur_s.copy()
            trial_s = _grow_all_to_convergence(new_c, trial_s, tol)
            if _is_valid(new_c, trial_s, tol):
                new_obj = _objective(new_c, trial_s)
                delta = new_obj - cur_obj
                if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
                    cur_c = new_c
                    cur_s = trial_s
                    cur_obj = new_obj
                    if cur_obj > best_obj:
                        best_obj = cur_obj
                        best_c = cur_c.copy()
                        best_s = cur_s.copy()
    return best_c, best_s


def _final_polish(centers, sides, tol=1e-7):
    """High-precision polish: many rounds of grow + micro-perturb."""
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(50):
        new_sides = _grow_all_to_convergence(centers, sides.copy(), tol, max_iters=300)
        if _is_valid(centers, new_sides, tol):
            sides = new_sides
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_sides = sides.copy()
                best_centers = centers.copy()
        
        rng = np.random.default_rng(seed=rnd * 7 + 13)
        for i in range(n):
            for _ in range(4):
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                new_c = centers.copy()
                new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i] / 2, 1.0 - sides[i] / 2)
                new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i] / 2, 1.0 - sides[i] / 2)
                if _is_valid(new_c, sides, tol):
                    trial_s = _grow_all_to_convergence(new_c, sides.copy(), tol, max_iters=200)
                    if _is_valid(new_c, trial_s, tol):
                        obj = _objective(new_c, trial_s)
                        if obj > best_obj + 1e-10:
                            best_obj = obj
                            best_sides = trial_s.copy()
                            best_centers = new_c.copy()
                            centers = new_c.copy()
                            sides = trial_s.copy()
    
    return best_centers, best_sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0

    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
        _init_heterogeneous_8(),
        _init_heterogeneous_9(),
        _init_heterogeneous_10(),
    ]

    best_centers = None
    best_sides = None
    best_obj = -1.0

    for centers, sides in inits:
        if time.time() > deadline - 30:
            break
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides):
            for _ in range(15):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        if not _is_valid(centers, sides):
            continue
        opt_centers, opt_sides = _local_optimize(centers, sides)
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()

    if time.time() < deadline - 80:
        for seed in range(25):
            if time.time() > deadline - 50:
                break
            centers, sides = _init_random(seed, n_large=2)
            centers = _clamp_to_unit(centers, sides)
            centers = _resolve_overlap(centers, sides)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
            opt_c, opt_s = _simulated_annealing(centers, sides, seed=seed)
            if _is_valid(opt_c, opt_s):
                obj = _objective(opt_c, opt_s)
                if obj > best_obj:
                    best_obj = obj
                    best_centers = opt_c.copy()
                    best_sides = opt_s.copy()

    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        if time.time() < deadline - 10:
            pc, ps = _final_polish(best_centers, best_sides)
            if _is_valid(pc, ps):
                if _objective(pc, ps) > _objective(best_centers, best_sides):
                    best_centers = pc
                    best_sides = ps
        final_sides = best_sides.copy()
        final_sides = _grow_all_to_convergence(best_centers, final_sides, tol=1e-7, max_iters=500)
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides

    return best_centers, best_sides


# EVOLVE-BLOCK-END
