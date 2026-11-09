import numpy as np
import time


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
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
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        # 2 large squares at top
        [0.25, 0.75],  # large 1: [0,0.5]x[0.5,1]
        [0.75, 0.75],  # large 2: [0.5,1]x[0.5,1]
        # 8 small squares in 2x4 grid below
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _local_refine(centers, sides, tol=1e-7, max_iter=2000):
    """
    Local refinement: try to grow squares while maintaining feasibility.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    deltas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for delta in deltas:
                new_sides = best_sides.copy()
                new_sides[i] = best_sides[i] + delta
                new_centers = best_centers.copy()
                
                s = new_sides[i]
                cx = np.clip(best_centers[i, 0], s/2, 1 - s/2)
                cy = np.clip(best_centers[i, 1], s/2, 1 - s/2)
                new_centers[i] = [cx, cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = np.sum(best_sides)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_sides


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Primary packing: 2 large (0.5) + 8 small (0.25), sum = 3
    c1, s1 = _make_2big_8small()
    if _is_feasible(c1, s1):
        best_centers = c1.copy()
        best_sides = s1.copy()
        best_val = np.sum(s1)
    
    # Local refinement
    if best_centers is not None and time.time() < deadline - 10:
        try:
            refined_centers, refined_sides = _local_refine(
                best_centers, best_sides, max_iter=2000
            )
            if _is_feasible(refined_centers, refined_sides):
                refined_val = np.sum(refined_sides)
                if refined_val > best_val:
                    best_val = refined_val
                    best_centers = refined_centers
                    best_sides = refined_sides
        except Exception:
            pass
    
    if best_centers is None:
        # Fallback: should not happen
        best_centers, best_sides = _make_2big_8small()
    
    return best_centers, best_sides
