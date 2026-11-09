# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    deadline = time.time() + 570  # Safety margin

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible (vectorized)."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            return False
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _sum_sides(sides):
        return np.sum(sides)

    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2*n] = centers.flatten()
        x[2*n:] = sides
        return x

    def _from_x(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def _penalized_objective(x):
        """
        Objective function for scipy minimize.
        Minimizes: -sum(sides) + penalty(infeasibility)
        """
        centers, sides = _from_x(x)
        
        # Ensure sides are positive for calculation stability
        if np.any(sides <= 0):
            return 1e6

        penalty = 0.0
        
        # Boundary penalties
        # x boundaries
        left = centers[:, 0] - sides / 2
        right = centers[:, 0] + sides / 2
        if np.any(left < 0):
            penalty += np.sum((left[left < 0]) ** 2) * 10
        if np.any(right > 1):
            penalty += np.sum((right[right > 1] - 1) ** 2) * 10
            
        # y boundaries
        bottom = centers[:, 1] - sides / 2
        top = centers[:, 1] + sides / 2
        if np.any(bottom < 0):
            penalty += np.sum((bottom[bottom < 0]) ** 2) * 10
        if np.any(top > 1):
            penalty += np.sum((top[top > 1] - 1) ** 2) * 10

        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                min_dx = (sides[i] + sides[j]) / 2
                min_dy = (sides[i] + sides[j]) / 2
                
                # If overlapping in both dimensions
                if dx < min_dx and dy < min_dy:
                    # Quadratic penalty for penetration depth
                    penalty += ((min_dx - dx) ** 2 + (min_dy - dy) ** 2) * 10

        return -np.sum(sides) + penalty

    # Initial Configuration: 2 large (0.5) + 8 small (0.25)
    # Large squares at bottom
    # Small squares in a 4x2 grid on top
    
    def get_initial_config_3_0():
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        
        # Square 0: Bottom Left Large
        centers[0] = [0.25, 0.25]
        sides[0] = 0.5
        
        # Square 1: Bottom Right Large
        centers[1] = [0.75, 0.25]
        sides[1] = 0.5
        
        # Squares 2-9: Top Grid 4x2
        # y centers: 0.625, 0.875
        # x centers: 0.125, 0.375, 0.625, 0.875
        idx = 2
        for row in range(2):
            for col in range(4):
                centers[idx, 0] = 0.125 + col * 0.25
                centers[idx, 1] = 0.625 + row * 0.25
                sides[idx] = 0.25
                idx += 1
                
        return centers, sides

    best_centers, best_sides = get_initial_config_3_0()
    best_obj = _sum_sides(best_sides)
    
    # Verify initial config
    if not _is_feasible(best_centers, best_sides):
        # Fallback to grid if initial is somehow invalid (shouldn't happen)
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        best_obj = _sum_sides(best_sides)

    # Optimization Loop
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with perturbations
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the current best or the base config
        if r == 0:
            c0, s0 = best_centers.copy(), best_sides.copy()
        else:
            c0, s0 = get_initial_config_3_0()
            # Add random noise
            c0 += rng.normal(0, 0.05, (n, 2))
            s0 += rng.normal(0, 0.05, n)
            s0 = np.clip(s0, 0.1, 1.0)
            # Clip centers to valid range based on sides
            for i in range(n):
                c0[i, 0] = np.clip(c0[i, 0], s0[i]/2, 1 - s0[i]/2)
                c0[i, 1] = np.clip(c0[i, 1], s0[i]/2, 1 - s0[i]/2)

        x0 = _to_x(c0, s0)
        
        # Bounds for L-BFGS-B
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
        
        try:
            res = minimize(_penalized_objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success:
                c_res, s_res = _from_x(res.x)
                
                # Check feasibility of the result
                if _is_feasible(c_res, s_res):
                    obj_res = _sum_sides(s_res)
                    if obj_res > best_obj:
                        best_centers = c_res.copy()
                        best_sides = s_res.copy()
                        best_obj = obj_res
        except Exception:
            pass

    # Final Polish: Local Search
    # Simple hill climbing on the best solution found
    step = 0.01
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for param in ['cx', 'cy', 's']:
                for delta in [-step, step]:
                    c_new = best_centers.copy()
                    s_new = best_sides.copy()
                    
                    if param == 'cx':
                        c_new[i, 0] += delta
                    elif param == 'cy':
                        c_new[i, 1] += delta
                    else:
                        s_new[i] = max(s_new[i] + delta, 0.01)
                        
                    # Clip to boundaries
                    s = s_new[i]
                    c_new[i, 0] = np.clip(c_new[i, 0], s/2, 1 - s/2)
                    c_new[i, 1] = np.clip(c_new[i, 1], s/2, 1 - s/2)
                    
                    if _is_feasible(c_new, s_new):
                        obj_new = _sum_sides(s_new)
                        if obj_new > best_obj + 1e-10:
                            best_centers = c_new
                            best_sides = s_new
                            best_obj = obj_new
                            improved = True
        step *= 0.9

    return best_centers, best_sides

# EVOLVE-BLOCK-END
