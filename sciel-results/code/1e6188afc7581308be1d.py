import numpy as np
import time


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _sum_sides(sides):
    """Objective: sum of side lengths."""
    return np.sum(sides)


def _penalty(centers, sides, tol=1e-7, weight=1e4):
    """Penalty for constraint violations."""
    n = len(sides)
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += weight * (1.0 - s)
        if cx - s/2 < -tol:
            pen += weight * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += weight * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += weight * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += weight * ((cy + s/2) - (1 + tol))
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            # Penalty if overlapping
            if abs(dx) < t and abs(dy) < t:
                pen += weight * ((t - abs(dx)) + (t - abs(dy)))
    return pen


def _objective(centers, sides, tol=1e-7, weight=1e4):
    """Combined objective: maximize sum of sides minus penalty."""
    return _sum_sides(sides) - _penalty(centers, sides, tol, weight)


def _make_2big_8small():
    """Create initial config: 2 large squares in top row, 8 small in bottom 2 rows."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 2 big squares in top: each 0.5 x 0.5
    s_big = 0.5
    centers[0] = [0.25, 0.75]
    centers[1] = [0.75, 0.75]
    sides[0] = s_big
    sides[1] = s_big
    
    # 8 small squares in bottom: 4 per row, each 0.25 x 0.25
    s_small = 0.25
    for i in range(4):
        # Row 1 (y=0.375)
        centers[2+i] = [(i+0.5)*s_small, 0.375]
        sides[2+i] = s_small
        # Row 2 (y=0.125)
        centers[6+i] = [(i+0.5)*s_small, 0.125]
        sides[6+i] = s_small
    
    return centers, sides


def _local_optimize(centers, sides, max_iter=5000, tol=1e-7):
    """Coordinate descent local optimization."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(centers, sides, tol)
    
    step_size = 0.01
    for iteration in range(max_iter):
        improved = False
        # Try adjusting each center and side
        for i in range(n):
            # Try adjusting cx
            for delta in [step_size, -step_size]:
                new_cx = centers[i, 0] + delta
                new_centers = centers.copy()
                new_centers[i, 0] = new_cx
                new_obj = _objective(new_centers, sides, tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            # Try adjusting cy
            for delta in [step_size, -step_size]:
                new_cy = centers[i, 1] + delta
                new_centers = centers.copy()
                new_centers[i, 1] = new_cy
                new_obj = _objective(new_centers, sides, tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            # Try adjusting side
            for delta in [step_size, -step_size]:
                new_s = sides[i] + delta
                if new_s <= 0:
                    continue
                new_sides = sides.copy()
                new_sides[i] = new_s
                new_obj = _objective(centers, new_sides, tol)
                if new_obj > best_obj and _is_feasible(centers, new_sides, tol):
                    best_centers = centers.copy()
                    best_sides = new_sides
                    best_obj = new_obj
                    sides = new_sides
                    improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_centers, best_sides


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize the sum of side lengths.
    Uses a 2-big-8-small initial configuration with local optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Initial configuration
    centers, sides = _make_2big_8small()
    
    # Verify initial feasibility
    if not _is_feasible(centers, sides):
        # Fallback to grid
        n = 10
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
    
    # Local optimization
    centers, sides = _local_optimize(centers, sides, max_iter=2000)
    
    # Check time
    if time.time() > deadline:
        return centers, sides
    
    # Try a few more optimization passes with different step sizes
    for _ in range(3):
        if time.time() > deadline:
            break
        centers, sides = _local_optimize(centers, sides, max_iter=1000)
    
    # Final feasibility check
    if not _is_feasible(centers, sides):
        # Return the grid fallback
        n = 10
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
    
    return centers, sides
