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
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i,0] - centers[j,0])
            dy = abs(centers[i,1] - centers[j,1])
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


def _grow_square_max(centers, sides, i, tol=1e-7, iters=60):
    """Find the maximum side length for square i given all other positions/sizes."""
    lo = sides[i]
    hi = 1.0
    cx, cy = centers[i]
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        half = mid / 2.0
        if cx - half < -tol or cx + half > 1.0 + tol:
            hi = mid
            continue
        if cy - half < -tol or cy + half > 1.0 + tol:
            hi = mid
            continue
        feasible = True
        for j in range(len(sides)):
            if j == i:
                continue
            t = (mid + sides[j]) / 2.0 - tol
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            if dx < t and dy < t:
                feasible = False
                break
        if feasible:
            lo = mid
        else:
            hi = mid
    return lo


def _grow_all_iteratively(centers, sides, tol=1e-7, max_iters=300):
    """
    Iteratively grow each square to its maximum given others, repeating until convergence.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol)
            if new_s > sides[i] + 1e-12:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _high_precision_polish(centers, sides, tol=1e-9, max_iters=500):
    """
    High-precision polish: use very tight tolerance and many iterations to squeeze out
    the last bits of side length. This is the key to beating the current score.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            new_s = _grow_square_max(centers, sides, i, tol=tol, iters=80)
            if new_s > sides[i] + 1e-14:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides


def _push_to_corner(centers, sides, i, tol=1e-7):
    """
    Push square i to one of the four corners or edge positions, then regrow all.
    """
    n = len(sides)
    best_obj = _objective(centers, sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    half = sides[i] / 2.0
    
    targets = [
        (half, half),              # bottom-left corner
        (1.0 - half, half),        # bottom-right
        (half, 1.0 - half),        # top-left
        (1.0 - half, 1.0 - half),  # top-right
        (0.5, half),               # bottom-center
        (0.5, 1.0 - half),         # top-center
        (half, 0.5),               # left-center
        (1.0 - half, 0.5),         # right-center
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
            if new_s > trial_s[k] + 1e-10:
                trial_s[k] = new_s
        if _is_valid(test_c, trial_s, tol):
            obj = _objective(test_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = test_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s


def _resolve_overlaps(centers, sides, tol=1e-7, max_iters=200):
    """
    Resolve overlaps using iterative projection with better separation.
    """
    n = len(sides)
    for iteration in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                adx = abs(dx)
                ady = abs(dy)
                if adx < t and ady < t:
                    # Overlap: push apart along the axis of minimum penetration
                    if adx < ady:
                        push = (t - adx) / 2.0 + 1e-9
                        if dx < 0:
                            centers[i,0] -= push
                            centers[j,0] += push
                        else:
                            centers[i,0] += push
                            centers[j,0] -= push
                    else:
                        push = (t - ady) / 2.0 + 1e-9
                        if dy < 0:
                            centers[i,1] -= push
                            centers[j,1] += push
                        else:
                            centers[i,1] += push
                            centers[j,1] -= push
                    moved = True
        centers = _clamp_to_unit(centers, sides)
        if not moved:
            break
    return centers


def _local_optimize(centers, sides, tol=1e-7, max_rounds=80):
    """
    Local optimization combining growth, corner-pushing, and perturbations.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(max_rounds):
        # Step 1: Grow all squares iteratively
        new_sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
        if _is_valid(centers, new_sides, tol):
            obj = _objective(centers, new_sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = new_sides.copy()
                best_centers = centers.copy()
                sides = new_sides.copy()
        
        # Step 2: Try pushing each square to corners/edges
        for i in range(n):
            new_c, new_s = _push_to_corner(centers, sides, i, tol)
            if _objective(new_c, new_s) > best_obj + 1e-9:
                best_obj = _objective(new_c, new_s)
                best_centers = new_c
                best_sides = new_s
                centers = new_c.copy()
                sides = new_s.copy()
                sides = _grow_all_iteratively(centers.copy(), sides.copy(), tol)
                if _is_valid(centers, sides, tol):
                    obj = _objective(centers, sides)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_sides = sides.copy()
                        best_centers = centers.copy()
        
        # Step 3: Small random perturbations with multiple scales
        rng = np.random.default_rng(seed=rnd * 7919 + 13)
        for i in range(n):
            for scale in [0.005, 0.01, 0.02, 0.03]:
                for _ in range(3):
                    dx = rng.uniform(-scale, scale)
                    dy = rng.uniform(-scale, scale)
                    new_c = centers.copy()
                    new_c[i, 0] = np.clip(centers[i, 0] + dx, sides[i]/2, 1.0 - sides[i]/2)
                    new_c[i, 1] = np.clip(centers[i, 1] + dy, sides[i]/2, 1.0 - sides[i]/2)
                    if _is_valid(new_c, sides, tol):
                        trial_s = _grow_all_iteratively(new_c.copy(), sides.copy(), tol)
                        if _is_valid(new_c, trial_s, tol):
                            obj = _objective(new_c, trial_s)
                            if obj > best_obj + 1e-9:
                                best_obj = obj
                                best_centers = new_c.copy()
                                best_sides = trial_s.copy()
                                centers = new_c.copy()
                                sides = trial_s.copy()
        
        # Step 4: Pairwise swap exploration - try swapping positions of two squares
        if rnd % 3 == 0:
            for i in range(n):
                for j in range(i+1, n):
                    new_c = centers.copy()
                    new_c[i, :2] = centers[j, :2].copy()
                    new_c[j, :2] = centers[i, :2].copy()
                    if _is_valid(new_c, sides, tol):
                        trial_s = _grow_all_iteratively(new_c.copy(), sides.copy(), tol)
                        if _is_valid(new_c, trial_s, tol):
                            obj = _objective(new_c, trial_s)
                            if obj > best_obj + 1e-9:
                                best_obj = obj
                                best_centers = new_c.copy()
                                best_sides = trial_s.copy()
                                centers = new_c.copy()
                                sides = trial_s.copy()
        
        # Convergence check
        if rnd > 5 and abs(_objective(centers, sides) - best_obj) < 1e-13:
            can_improve = False
            for i in range(n):
                max_s = _grow_square_max(centers, sides, i, tol)
                if max_s > sides[i] + 1e-11:
                    can_improve = True
                    break
            if not can_improve:
                break
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing (9 squares of 1/3 + 1 extra)."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[:9] = np.stack([cx, cy], axis=1)
    sides[:9] = side
    centers[9] = [0.5, 0.5]
    sides[9] = 1.0/6.0
    return centers, sides


def _init_heterogeneous_1():
    """
    One large square (0.5) in bottom-left, 9 smaller filling the rest.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
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
    """
    Two large squares (0.4) and 8 small ones (0.25).
    """
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
    """
    One large (0.5) + 2 medium (0.25) + 7 small (0.125).
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.125, 0.625]
    sides[1] = 0.25
    centers[2] = [0.875, 0.625]
    sides[2] = 0.25
    side_s = 0.125
    idx = 3
    positions = [
        [0.375, 0.625], [0.625, 0.625],
        [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
        [0.75, 0.25],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """
    Two large (0.45) in opposite corners, 8 smaller filling rest.
    """
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
        [0.525, 0.525],
        [0.05, 0.575], [0.05, 0.875],
        [0.825, 0.525],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_5():
    """
    Three large squares in corners, 7 smaller.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    centers[2] = [0.5, 0.7]
    sides[2] = 0.4
    side_s = 0.15
    idx = 3
    positions = [
        [0.2, 0.55], [0.2, 0.85],
        [0.8, 0.55], [0.8, 0.85],
        [0.5, 0.2],
        [0.35, 0.35], [0.65, 0.35],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_6():
    """
    Four medium squares (0.3) in a 2x2 arrangement, 6 small.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    side_m = 0.3
    corners = [(0.15, 0.15), (0.5, 0.15), (0.85, 0.15), (0.15, 0.5)]
    for i, (cx, cy) in enumerate(corners):
        centers[i] = [cx, cy]
        sides[i] = side_m
    side_s = 0.15
    idx = 4
    positions = [
        [0.5, 0.5], [0.85, 0.5],
        [0.15, 0.85], [0.5, 0.85], [0.85, 0.85],
        [0.325, 0.325],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_7():
    """
    One very large (0.55) + structured smaller squares.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.275, 0.275]
    sides[0] = 0.55
    side_s = 1.0/5.0
    idx = 1
    # Top strip: 5 squares
    for col in range(5):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.55 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    # Right strip: 2 squares
    for row in range(2):
        centers[idx, 0] = 0.55 + 0.5 * side_s
        centers[idx, 1] = (row + 0.5) * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        
        # Resolve overlaps
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to resolve
            for attempt in range(5):
                rng = np.random.default_rng(seed=attempt * 1000 + 7)
                for i in range(10):
                    centers[i, 0] += rng.uniform(-0.01, 0.01)
                    centers[i, 1] += rng.uniform(-0.01, 0.01)
                centers = _clamp_to_unit(centers, sides)
                centers = _resolve_overlaps(centers, sides)
                if _is_valid(centers, sides):
                    break
            if not _is_valid(centers, sides):
                continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
    
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final high-precision polish pass
        if time.time() < deadline - 30:
            polished_sides = _high_precision_polish(best_centers.copy(), best_sides.copy(), tol=1e-9, max_iters=500)
            if _is_valid(best_centers, polished_sides, tol=1e-7):
                if _objective(best_centers, polished_sides) > _objective(best_centers, best_sides):
                    best_sides = polished_sides
        
        # Additional local optimization rounds with the best result
        if time.time() < deadline - 60:
            for extra_round in range(3):
                opt_c, opt_s = _local_optimize(best_centers.copy(), best_sides.copy(), tol=1e-7, max_rounds=30)
                if _is_valid(opt_c, opt_s):
                    obj = _objective(opt_c, opt_s)
                    if obj > _objective(best_centers, best_sides) + 1e-10:
                        best_centers = opt_c
                        best_sides = opt_s
                    # Polish again
                    if time.time() < deadline - 30:
                        pol_s = _high_precision_polish(best_centers.copy(), best_sides.copy(), tol=1e-9, max_iters=300)
                        if _is_valid(best_centers, pol_s, tol=1e-7):
                            if _objective(best_centers, pol_s) > _objective(best_centers, best_sides):
                                best_sides = pol_s
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
