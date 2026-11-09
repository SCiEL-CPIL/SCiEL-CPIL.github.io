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
    deadline = time.time() + 570  # safety margin before 600s limit

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible."""
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

    def _objective(centers, sides):
        """Maximize sum of sides, penalize infeasibility."""
        if _is_feasible(centers, sides):
            return _sum_sides(sides)
        return -1e6

    def _softplus(x):
        """Numerically stable softplus for smooth penalties."""
        return np.where(x > 30, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))

    def _smooth_penalty(x):
        """
        Smooth penalty function for scipy minimize.
        x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
        Uses softplus to provide continuous gradients.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Boundary violations: push centers inward
        penalty = np.zeros(1)
        # Left boundary: cx - s/2 >= 0
        penalty[0] += np.sum(_softplus(-(centers[:, 0] - sides/2)))
        # Right boundary: cx + s/2 <= 1
        penalty[0] += np.sum(_softplus(centers[:, 0] + sides/2 - 1))
        # Bottom boundary: cy - s/2 >= 0
        penalty[0] += np.sum(_softplus(-(centers[:, 1] - sides/2)))
        # Top boundary: cy + s/2 <= 1
        penalty[0] += np.sum(_softplus(centers[:, 1] + sides/2 - 1))
        
        # Overlap violations: for each pair, |cx_i - cx_j| >= (s_i+s_j)/2 OR |cy_i - cy_j| >= (s_i+s_j)/2
        # We penalize when both are violated
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                # Violation amounts
                vx = t - dx  # positive if x-overlap
                vy = t - dy  # positive if y-overlap
                # We want at least one of vx <= 0 or vy <= 0
                # Penalty when both are positive
                penalty[0] += _softplus(vx) + _softplus(vy) - _softplus(vx + vy)
        
        # Positive side penalty
        penalty[0] += np.sum(_softplus(-sides))
        
        return penalty[0]

    def _scipy_objective(x):
        """Combined objective: maximize sum of sides, penalize infeasibility with smooth penalties."""
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Ensure sides are reasonably positive for the objective
        if np.any(sides <= 0.001):
            return 1e6
        
        sum_sides = np.sum(sides)
        pen = _smooth_penalty(x)
        
        # Weighted combination
        return -sum_sides + 100.0 * pen

    # Start with the baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)

    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2*n] = centers.flatten()
        x[2*n:] = sides
        return x

    def _from_x(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    # Phase 1: Local search with random perturbations
    rng = np.random.default_rng(42)
    step_size = 0.05
    max_iters = 20000
    
    current_x = _to_x(centers, sides)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random perturbation
        x_p = current_x + rng.normal(0, step_size, 3 * n)
        centers_p, sides_p = _from_x(x_p)
        sides_p = np.maximum(sides_p, 0.01)  # keep sides positive
        
        # Simple repair: clip to boundaries
        for i in range(n):
            s = sides_p[i]
            centers_p[i, 0] = np.clip(centers_p[i, 0], s/2, 1 - s/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], s/2, 1 - s/2)
        
        x_p = _to_x(centers_p, sides_p)
        obj_p = _objective(centers_p, sides_p)
        if obj_p > best_obj:
            best_centers = centers_p.copy()
            best_sides = sides_p.copy()
            best_obj = obj_p
            current_x = x_p
            step_size *= 0.95
        else:
            if rng.random() < 0.1:
                current_x = x_p
                step_size *= 1.05
        
        if it % 100 == 0:
            current_x = _to_x(best_centers, best_sides)

    # Phase 2: Use scipy with smooth penalties to polish from multiple starting points
    if time.time() < deadline - 60:
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.01, 1.0))
        
        # Start from best solution found so far
        starting_points = [_to_x(best_centers, best_sides)]
        
        # Also try some random starts
        for _ in range(5):
            if time.time() > deadline - 60:
                break
            c_rand = np.random.uniform(0.1, 0.9, (n, 2))
            s_rand = np.random.uniform(0.08, 0.25, n)
            # Clip to boundaries
            for i in range(n):
                s = s_rand[i]
                c_rand[i, 0] = np.clip(c_rand[i, 0], s/2, 1 - s/2)
                c_rand[i, 1] = np.clip(c_rand[i, 1], s/2, 1 - s/2)
            starting_points.append(_to_x(c_rand, s_rand))
        
        for x0 in starting_points:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(_scipy_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12,
                                       'maxls': 50, 'maxfun': 5000})
                c_res, s_res = _from_x(res.x)
                # Check feasibility
                if _is_feasible(c_res, s_res):
                    obj_res = _sum_sides(s_res)
                    if obj_res > best_obj:
                        best_centers = c_res.copy()
                        best_sides = s_res.copy()
                        best_obj = obj_res
            except Exception:
                pass

    # Phase 3: Final high-precision polish with coordinate descent
    for _ in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for dx in [-0.0005, 0.0005]:
                for dy in [-0.0005, 0.0005]:
                    for ds in [-0.0005, 0.0005]:
                        c_new = best_centers.copy()
                        s_new = best_sides.copy()
                        c_new[i, 0] += dx
                        c_new[i, 1] += dy
                        s_new[i] = max(s_new[i] + ds, 0.01)
                        s = s_new[i]
                        c_new[i, 0] = np.clip(c_new[i, 0], s/2, 1 - s/2)
                        c_new[i, 1] = np.clip(c_new[i, 1], s/2, 1 - s/2)
                        
                        obj_new = _objective(c_new, s_new)
                        if obj_new > best_obj + 1e-10:
                            best_centers = c_new
                            best_sides = s_new
                            best_obj = obj_new
                            improved = True
        if not improved:
            break

    # Final feasibility check and return
    if not _is_feasible(best_centers, best_sides):
        # Fall back to grid packing
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
