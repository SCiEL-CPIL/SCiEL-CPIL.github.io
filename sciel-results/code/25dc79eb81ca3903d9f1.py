# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 570.0  # safety margin
    n = 10

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i] / 2.0 < -tol or cx[i] + s[i] / 2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i] / 2.0 < -tol or cy[i] + s[i] / 2.0 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def _score(centers, sides):
        if not _is_feasible(centers, sides):
            return -np.inf
        return np.sum(sides)

    def _make_layout_30():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        # Row 0: 4 squares of 0.25
        s1 = 0.25
        for j in range(4):
            centers[idx, 0] = j * s1 + s1 / 2
            centers[idx, 1] = s1 / 2
            sides[idx] = s1
            idx += 1
        # Row 1: 3 squares of 1/3
        s2 = 1.0 / 3.0
        y1 = s1 + s2 / 2
        for j in range(3):
            centers[idx, 0] = j * s2 + s2 / 2
            centers[idx, 1] = y1
            sides[idx] = s2
            idx += 1
        # Row 2: 3 squares of 1/3
        s3 = 1.0 / 3.0
        y2 = s1 + s2 + s3 / 2
        for j in range(3):
            centers[idx, 0] = j * s3 + s3 / 2
            centers[idx, 1] = y2
            sides[idx] = s3
            idx += 1
        return centers, sides

    best_centers, best_sides = _make_layout_30()
    best_score = _score(best_centers, best_sides)

    # Define objective function for scipy minimize
    # We minimize the negative sum of sides, plus a penalty for constraint violations.
    def _objective(params):
        # params: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
        centers = params[0::3].reshape(-1, 2)
        sides = params[1::3]
        
        # Penalty for being out of bounds
        penalty = 0.0
        for i in range(n):
            if centers[i, 0] - sides[i] / 2.0 < 0:
                penalty += (0 - (centers[i, 0] - sides[i] / 2.0)) ** 2
            if centers[i, 0] + sides[i] / 2.0 > 1.0:
                penalty += ((centers[i, 0] + sides[i] / 2.0) - 1.0) ** 2
            if centers[i, 1] - sides[i] / 2.0 < 0:
                penalty += (0 - (centers[i, 1] - sides[i] / 2.0)) ** 2
            if centers[i, 1] + sides[i] / 2.0 > 1.0:
                penalty += ((centers[i, 1] + sides[i] / 2.0) - 1.0) ** 2
            if sides[i] <= 0:
                penalty += sides[i] ** 2
        
        # Penalty for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2.0
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                # If |dx| < t and |dy| < t, they overlap
                # We want to push them apart
                if abs(dx) < t:
                    penalty += (t - abs(dx)) ** 2
                if abs(dy) < t:
                    penalty += (t - abs(dy)) ** 2
        
        return -np.sum(sides) + 100.0 * penalty

    # Flatten initial solution
    def _flatten(centers, sides):
        params = np.zeros(3 * n)
        for i in range(n):
            params[3 * i] = centers[i, 0]
            params[3 * i + 1] = centers[i, 1]
            params[3 * i + 2] = sides[i]
        return params

    def _unflatten(params):
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            centers[i, 0] = params[3 * i]
            centers[i, 1] = params[3 * i + 1]
            sides[i] = params[3 * i + 2]
        return centers, sides

    # Run scipy minimize
    if time.time() < DEADLINE - 100:
        initial_params = _flatten(best_centers, best_sides)
        try:
            res = minimize(
                _objective,
                initial_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            new_centers, new_sides = _unflatten(res.x)
            # Clip sides to be positive
            new_sides = np.clip(new_sides, 1e-6, 0.99)
            # Re-center if necessary
            for i in range(n):
                new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
                new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
            
            if _is_feasible(new_centers, new_sides):
                new_score = _score(new_centers, new_sides)
                if new_score > best_score:
                    best_centers = new_centers
                    best_sides = new_sides
                    best_score = new_score
        except Exception as e:
            pass

    # Try a few random restarts with scipy polish
    np.random.seed(42)
    for _ in range(5):
        if time.time() > DEADLINE - 60:
            break
        centers = np.random.rand(n, 2)
        sides = np.random.rand(n) * 0.2 + 0.1
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
            centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
        
        # Quick feasibility fix
        for _ in range(50):
            if _is_feasible(centers, sides):
                break
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2 - TOL
                    if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                        sides[i] *= 0.9
                        sides[j] *= 0.9
                        centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
                        centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
                        centers[j, 0] = np.clip(centers[j, 0], sides[j]/2, 1 - sides[j]/2)
                        centers[j, 1] = np.clip(centers[j, 1], sides[j]/2, 1 - sides[j]/2)
        
        if _is_feasible(centers, sides):
            initial_params = _flatten(centers, sides)
            try:
                res = minimize(
                    _objective,
                    initial_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                new_centers, new_sides = _unflatten(res.x)
                new_sides = np.clip(new_sides, 1e-6, 0.99)
                for i in range(n):
                    new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
                    new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
                
                if _is_feasible(new_centers, new_sides):
                    new_score = _score(new_centers, new_sides)
                    if new_score > best_score:
                        best_centers = new_centers
                        best_sides = new_sides
                        best_score = new_score
            except Exception as e:
                pass

    if not _is_feasible(best_centers, best_sides):
        best_centers, best_sides = _make_layout_30()
        best_score = _score(best_centers, best_sides)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
