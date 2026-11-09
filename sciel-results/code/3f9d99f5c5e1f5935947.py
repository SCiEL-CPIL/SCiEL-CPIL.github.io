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
    # Inside [-tol, 1+tol]^2
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
            return False
    # Non-overlap: for i < j, must not have |cx_i-cx_j| < t AND |cy_i-cy_j| < t
    # where t = (s_i + s_j)/2 - tol
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


def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    Returns modified centers, sides.
    """
    n = len(sides)
    for iteration in range(200):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Overlapping: push apart along the axis of smaller gap
                    if abs(dx) < abs(dy):
                        # Push horizontally
                        push = (t - abs(dx)) / 2.0 + tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        # Push vertically
                        push = (t - abs(dy)) / 2.0 + tol
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


def _try_grow_squares(centers, sides, tol=1e-7, max_iters=500):
    """
    Try to grow each square as much as possible while maintaining feasibility.
    Uses iterative bisection on each square's size.
    """
    n = len(sides)
    for iteration in range(max_iters):
        improved = False
        for i in range(n):
            # Try to grow square i
            lo = sides[i]
            hi = 1.0  # maximum possible side
            # Binary search for max size
            for _ in range(30):
                mid = (lo + hi) / 2.0
                test_sides = sides.copy()
                test_sides[i] = mid
                # Check if square i fits inside unit square
                half = mid / 2.0
                if centers[i, 0] - half < -tol or centers[i, 0] + half > 1.0 + tol:
                    hi = mid
                    continue
                if centers[i, 1] - half < -tol or centers[i, 1] + half > 1.0 + tol:
                    hi = mid
                    continue
                # Check non-overlap with all others
                feasible = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (mid + sides[j]) / 2.0 - tol
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            # If we can grow
            if lo > sides[i] + 1e-9:
                sides[i] = lo
                improved = True
        if not improved:
            break
    return sides


def _local_optimize(centers, sides, tol=1e-7, max_rounds=100):
    """
    Local optimization: alternate between growing squares and adjusting positions.
    """
    n = len(sides)
    best_sides = sides.copy()
    best_centers = centers.copy()
    best_obj = _objective(best_centers, best_sides)
    
    for rnd in range(max_rounds):
        # Try to grow all squares
        new_sides = sides.copy()
        for i in range(n):
            lo = new_sides[i]
            hi = 1.0
            for _ in range(25):
                mid = (lo + hi) / 2.0
                test_sides = new_sides.copy()
                test_sides[i] = mid
                half = mid / 2.0
                if centers[i, 0] - half < -tol or centers[i, 0] + half > 1.0 + tol:
                    hi = mid
                    continue
                if centers[i, 1] - half < -tol or centers[i, 1] + half > 1.0 + tol:
                    hi = mid
                    continue
                feasible = True
                for j in range(n):
                    if j == i:
                        continue
                    t = (mid + new_sides[j]) / 2.0 - tol
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            if lo > new_sides[i] + 1e-9:
                new_sides[i] = lo
        
        if _is_valid(centers, new_sides, tol):
            sides = new_sides
            obj = _objective(centers, sides)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sides = sides.copy()
                best_centers = centers.copy()
        
        # Try position adjustments: for each square, try moving it slightly
        # to allow others to grow
        for i in range(n):
            orig_cx = centers[i, 0]
            orig_cy = centers[i, 1]
            half = sides[i] / 2.0
            # Try a few position perturbations
            for dx in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                for dy in [-0.05, -0.02, 0.0, 0.02, 0.05]:
                    new_cx = orig_cx + dx
                    new_cy = orig_cy + dy
                    if new_cx - half < -tol or new_cx + half > 1.0 + tol:
                        continue
                    if new_cy - half < -tol or new_cy + half > 1.0 + tol:
                        continue
                    test_centers = centers.copy()
                    test_centers[i, 0] = new_cx
                    test_centers[i, 1] = new_cy
                    # Check validity
                    if _is_valid(test_centers, sides, tol):
                        # Try growing all squares from this position
                        trial_sides = sides.copy()
                        for k in range(n):
                            lo = trial_sides[k]
                            hi = 1.0
                            for _ in range(20):
                                mid = (lo + hi) / 2.0
                                test_s = trial_sides.copy()
                                test_s[k] = mid
                                h = mid / 2.0
                                if test_centers[k, 0] - h < -tol or test_centers[k, 0] + h > 1.0 + tol:
                                    hi = mid
                                    continue
                                if test_centers[k, 1] - h < -tol or test_centers[k, 1] + h > 1.0 + tol:
                                    hi = mid
                                    continue
                                feasible = True
                                for l in range(n):
                                    if l == k:
                                        continue
                                    t = (mid + trial_sides[l]) / 2.0 - tol
                                    ddx = abs(test_centers[k, 0] - test_centers[l, 0])
                                    ddy = abs(test_centers[k, 1] - test_centers[l, 1])
                                    if ddx < t and ddy < t:
                                        feasible = False
                                        break
                                if feasible:
                                    lo = mid
                                else:
                                    hi = mid
                            if lo > trial_sides[k] + 1e-9:
                                trial_sides[k] = lo
                        if _is_valid(test_centers, trial_sides, tol):
                            obj = _objective(test_centers, trial_sides)
                            if obj > best_obj + 1e-9:
                                best_obj = obj
                                best_centers = test_centers.copy()
                                best_sides = trial_sides.copy()
                                centers = test_centers.copy()
                                sides = trial_sides.copy()
    
    return best_centers, best_sides


def _init_grid_packing():
    """Baseline 3x3 grid packing."""
    n = 10
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _init_heterogeneous_1():
    """
    One large square (side 0.5) in bottom-left, rest in top-right region.
    Large square: [0, 0.5] x [0, 0.5], center (0.25, 0.25)
    Remaining region: [0,1]x[0.5,1] and [0.5,1]x[0,0.5]
    Place 9 squares in the top strip [0,1]x[0.5,1] (height 0.5)
    and right strip [0.5,1]x[0,0.5] (width 0.5).
    Actually let's put all 9 in the top-right L-shape.
    Simpler: put 9 squares in the region above and to the right.
    Let's use a 3x3 grid in [0,1]x[0.5,1] for 6, and [0.5,1]x[0,0.5] for 3.
    Top strip: 3 cols x 2 rows = 6 squares, side = 0.5/3 = 1/6
    Right strip below: 3 cols x 1 row = 3 squares, side = 0.5/3 = 1/6
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Square 0: large
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # Top strip [0,1]x[0.5,1]: 3 cols, 2 rows
    side_t = 1.0 / 3.0
    idx = 1
    for row in range(2):
        for col in range(3):
            centers[idx, 0] = (col + 0.5) * side_t
            centers[idx, 1] = 0.5 + (row + 0.5) * side_t
            sides[idx] = side_t
            idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols, 1 row
    side_r = 1.0 / 3.0
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_r
        centers[idx, 1] = 0.25
        sides[idx] = side_r
        idx += 1
    return centers, sides


def _init_heterogeneous_2():
    """
    Two large squares (side 0.4) and 8 small ones.
    Square 0: [0, 0.4] x [0, 0.4], center (0.2, 0.2)
    Square 1: [0.6, 1.0] x [0, 0.4], center (0.8, 0.2)
    Remaining: top strip [0,1]x[0.4,1] (height 0.6) and middle [0.4,0.6]x[0,0.4]
    8 squares in top strip: 4 cols x 2 rows, side = 0.6/4 = 0.15... 
    Let's do 4x2 in top: side = 1/4 = 0.25? No, height is 0.6, so 2 rows of 0.3.
    4 cols of 0.25, 2 rows of 0.3 -> but squares must be square.
    Use side = 0.25: 4 cols fit in width 1.0, 2 rows need height 0.5 <= 0.6. OK.
    That's 8 squares of side 0.25 in top strip.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.2, 0.2]
    sides[0] = 0.4
    centers[1] = [0.8, 0.2]
    sides[1] = 0.4
    # 8 squares of side 0.25 in top strip [0,1]x[0.4, 0.9]
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
    One very large square (side 0.5) and 9 small squares filling the rest.
    Large: [0,0.5]x[0,0.5]
    Rest: L-shape. Fill with 9 squares of side ~0.167 in a 3x3 grid in [0,1]x[0.5,1]
    plus [0.5,1]x[0,0.5].
    Actually same as heterogeneous_1 but let's try different sizes.
    9 squares in the L: top strip [0,1]x[0.5,1] has area 0.5, right strip [0.5,1]x[0,0.5] area 0.25.
    Total L area = 0.75. 9 squares, avg area = 0.75/9 = 0.0833, side ~ 0.289.
    But they must fit geometrically. Let's just use 1/6 side (0.167) for all 9.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    # Top strip [0,1]x[0.5,1]: 6 cols x 1 row? side=1/6, 6 cols = 1.0, 1 row height 1/6 <= 0.5
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    # Right strip [0.5,1]x[0,0.5]: 3 cols x 1 row? side=1/6, 3 cols = 0.5, fits in width 0.5
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides


def _init_heterogeneous_4():
    """
    Try a packing with squares of varying sizes based on a known good configuration.
    Based on literature: for n=10, optimal sum is around 1.6 or higher.
    Let's try: 1 square of side 0.5, 2 of side 0.25, 7 of side ~0.14.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    # Large square bottom-left
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    # Two medium squares top
    centers[1] = [0.375, 0.75]
    sides[1] = 0.25
    centers[2] = [0.75, 0.75]
    sides[2] = 0.25
    # 7 small squares filling gaps
    # Gap regions: [0,0.5]x[0.5,1] left-top, [0.5,1]x[0,0.5] right-bottom, [0.5,0.5]x[0.5,1]... 
    # This is getting complex. Let's just place them and let optimizer fix.
    side_s = 0.14
    idx = 3
    positions = [
        [0.12, 0.62], [0.38, 0.62],
        [0.62, 0.12], [0.88, 0.12],
        [0.62, 0.38], [0.88, 0.38],
        [0.12, 0.88],
    ]
    for pos in positions:
        centers[idx] = pos
        sides[idx] = side_s
        idx += 1
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
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
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
