# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _count_convex7(points: np.ndarray) -> int:
    """
    Count the number of 7-point subsets in convex position.
    Uses brute force over all C(33,7) = 4,272,048 subsets.
    For each subset, check if the 7 points form a convex polygon
    by checking that all 7 turns are strictly convex (left turns).
    """
    n = len(points)
    count = 0
    # Precompute for speed
    for i in range(n - 6):
        xi, yi = points[i]
        for j in range(i + 1, n - 5):
            xj, yj = points[j]
            for k in range(j + 1, n - 4):
                xk, yk = points[k]
                for l in range(k + 1, n - 3):
                    xl, yl = points[l]
                    for m in range(l + 1, n - 2):
                        xm, ym = points[m]
                        for o in range(m + 1, n - 1):
                            xo, yo = points[o]
                            for p in range(o + 1, n):
                                xp, yp = points[p]
                                # Check 7 turns: (i,j,k), (j,k,l), (k,l,m), (l,m,o), (m,o,p), (o,p,i), (p,i,j)
                                # Turn at j: (xj-xi)*(yk-yj) - (yj-yi)*(xk-xj)
                                t1 = (xj-xi)*(yk-yj) - (yj-yi)*(xk-xj)
                                if t1 <= 1e-9:
                                    continue
                                # Turn at k: (xk-xj)*(yl-yk) - (yk-yj)*(xl-xk)
                                t2 = (xk-xj)*(yl-yk) - (yk-yj)*(xl-xk)
                                if t2 <= 1e-9:
                                    continue
                                # Turn at l: (xl-xk)*(ym-yl) - (yl-yk)*(xm-xl)
                                t3 = (xl-xk)*(ym-yl) - (yl-yk)*(xm-xl)
                                if t3 <= 1e-9:
                                    continue
                                # Turn at m: (xm-xl)*(yo-ym) - (ym-yl)*(xo-xm)
                                t4 = (xm-xl)*(yo-ym) - (ym-yl)*(xo-xm)
                                if t4 <= 1e-9:
                                    continue
                                # Turn at o: (xo-xm)*(yp-yo) - (yo-ym)*(xp-xo)
                                t5 = (xo-xm)*(yp-yo) - (yo-ym)*(xp-xo)
                                if t5 <= 1e-9:
                                    continue
                                # Turn at p: (xp-xo)*(yi-yp) - (yp-yo)*(xi-xp)
                                t6 = (xp-xo)*(yi-yp) - (yp-yo)*(xi-xp)
                                if t6 <= 1e-9:
                                    continue
                                # Turn at i: (xi-xp)*(yj-yi) - (yi-yp)*(xj-xi)
                                t7 = (xi-xp)*(yj-yi) - (yi-yp)*(xj-xi)
                                if t7 <= 1e-9:
                                    continue
                                count += 1
    return count

def _is_valid(points: np.ndarray) -> bool:
    """Check validity constraints."""
    n = len(points)
    if points.shape != (33, 2):
        return False
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False
    if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
        return False
    # Check minimum distance
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            if dx*dx + dy*dy < 1e-12:
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                val = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
                if abs(val) < 1e-9:
                    return False
    return True

def _generate_structured_initial(n=33):
    """
    Generate an initial configuration that clusters points near the boundary
    of the unit square to minimize convex 7-gons.
    
    Strategy: Place points along the edges of the square, creating a "ring"
    configuration where most points are near the boundary. This reduces the
    number of points that can form convex polygons because interior points
    are scarce.
    """
    np.random.seed(123)
    points = np.zeros((n, 2))
    
    # Place points along the 4 edges of the square
    # Bottom edge: y near 0
    n_bottom = 8
    # Top edge: y near 1
    n_top = 8
    # Left edge: x near 0
    n_left = 8
    # Right edge: x near 1
    n_right = 8
    # Interior: a few points
    n_interior = n - n_bottom - n_top - n_left - n_right
    
    idx = 0
    eps = 0.02
    
    # Bottom edge
    for i in range(n_bottom):
        points[idx, 0] = (i + 1) / (n_bottom + 1)
        points[idx, 1] = np.random.uniform(0, eps)
        idx += 1
    
    # Right edge
    for i in range(n_right):
        points[idx, 0] = np.random.uniform(1 - eps, 1)
        points[idx, 1] = (i + 1) / (n_right + 1)
        idx += 1
    
    # Top edge
    for i in range(n_top):
        points[idx, 0] = (i + 1) / (n_top + 1)
        points[idx, 1] = np.random.uniform(1 - eps, 1)
        idx += 1
    
    # Left edge
    for i in range(n_left):
        points[idx, 0] = np.random.uniform(0, eps)
        points[idx, 1] = (i + 1) / (n_left + 1)
        idx += 1
    
    # Interior points - place them near the center but slightly off
    for i in range(n_interior):
        points[idx, 0] = np.random.uniform(0.3, 0.7)
        points[idx, 1] = np.random.uniform(0.3, 0.7)
        idx += 1
    
    # Ensure validity by adding small perturbations if needed
    for _ in range(1000):
        if _is_valid(points):
            break
        # Add small random perturbation to break degeneracies
        points += np.random.uniform(-0.001, 0.001, size=(n, 2))
        points = np.clip(points, 0.001, 0.999)
    
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres problem: place 33 points in the unit square to minimize
    the number of convex 7-gons.
    """
    start_time = time.time()
    deadline = start_time + 615  # Safety margin before 630s kill
    
    n = 33
    
    # Generate structured initial configuration
    best_points = _generate_structured_initial(n)
    best_count = _count_convex7(best_points)
    best_value = 1.0 / (1.0 + best_count)
    
    # Simulated annealing with structured moves
    current_points = best_points.copy()
    current_count = best_count
    current_value = best_value
    
    T = 0.03  # Initial temperature
    T_min = 0.0005
    cooling = 0.9995
    
    iteration = 0
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(n)
        
        # Generate perturbation scaled with temperature
        scale = 0.01 * (T / 0.03)
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        
        old_x = current_points[idx, 0]
        old_y = current_points[idx, 1]
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Clamp to unit square with margin
        new_x = np.clip(new_x, 0.001, 0.999)
        new_y = np.clip(new_y, 0.001, 0.999)
        
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        # Check validity
        if not _is_valid(current_points):
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
            T *= cooling
            continue
        
        new_count = _count_convex7(current_points)
        new_value = 1.0 / (1.0 + new_count)
        
        delta = new_value - current_value
        if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
            current_count = new_count
            current_value = new_value
            if new_value > best_value:
                best_value = new_value
                best_points = current_points.copy()
        else:
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        iteration += 1
        # Periodic restart to escape local optima
        if iteration % 5000 == 0 and time.time() < deadline - 60:
            # Try a few random restarts from structured configurations
            for _ in range(3):
                restart_points = _generate_structured_initial(n)
                restart_count = _count_convex7(restart_points)
                restart_value = 1.0 / (1.0 + restart_count)
                if restart_value > best_value:
                    best_value = restart_value
                    best_points = restart_points.copy()
                    current_points = restart_points.copy()
                    current_count = restart_count
                    current_value = restart_value
                    break
    
    return best_points

# EVOLVE-BLOCK-END
