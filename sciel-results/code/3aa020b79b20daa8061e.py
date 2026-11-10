# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, QhullError
import itertools

def squares_in_square10() -> tuple:
    """
    Optimize placement of 10 axis-aligned squares in the unit square to
    maximize the sum of side lengths, with pairwise non-overlapping interiors.
    
    Strategy:
    1. Start from several diverse initial configurations (grid, random, etc.)
    2. For each, use a smooth penalty-based optimization where the side length
       of each square is determined by the Voronoi cell constraint (max possible
       given neighbors and boundaries).
    3. Use scipy's L-BFGS-B to optimize centers and maximize sum of sides.
    4. Return the best valid configuration found.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 580.0  # safety margin before 600s limit
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    def compute_max_sides(cx, cy):
        """
        Given centers, compute the maximum side length each square can have
        such that all squares remain inside [0,1]^2 and don't overlap.
        This is a conservative (valid) computation.
        """
        sides = np.zeros(n)
        for i in range(n):
            # Boundary constraints
            max_s = min(2.0 * cx[i], 2.0 * (1.0 - cx[i]),
                        2.0 * cy[i], 2.0 * (1.0 - cy[i]))
            # Neighbor constraints: for square i and j to not overlap,
            # |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
            # We enforce: s_i <= 2 * min over j of max(|dx|, |dy|) - s_j is tricky
            # Instead, use: s_i <= 2 * min_j max(|cx_i-cx_j|, |cy_i-cy_j|) - s_j
            # This is conservative. Better: s_i + s_j <= 2 * max(|dx|, |dy|)
            # So s_i <= 2 * max(|dx|, |dy|) - s_j for all j.
            # We'll iterate to find consistent sides.
            for j in range(n):
                if i == j:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                gap = max(dx, dy)
                # Constraint: (s_i + s_j) / 2 <= gap  =>  s_i <= 2*gap - s_j
                # For now, use a simple bound: s_i <= 2*gap - s_j
                # But s_j is unknown. Use a fixed-point style or just bound by gap.
                # Conservative: s_i <= 2 * gap - s_j. If we assume s_j <= some bound...
                # Simpler: s_i <= 2 * gap - s_j, and we'll solve iteratively below.
                # For now, just use s_i <= 2*gap - s_j with s_j from previous iteration.
                pass
            sides[i] = max(0.0, max_s)
        return sides
    
    def compute_sides_iterative(cx, cy, max_iter=50):
        """
        Iteratively compute maximum valid sides given centers.
        Uses the constraint: for each pair (i,j), s_i + s_j <= 2 * max(|dx_ij|, |dy_ij|)
        and boundary: s_i <= 2*min(cx_i, 1-cx_i, cy_i, 1-cy_i)
        """
        # Initialize with boundary-limited sides
        bound_s = np.array([min(2.0*cx[i], 2.0*(1.0-cx[i]), 2.0*cy[i], 2.0*(1.0-cy[i])) for i in range(n)])
        sides = np.maximum(bound_s, 0.0)
        
        for _ in range(max_iter):
            new_sides = sides.copy()
            for i in range(n):
                # Boundary constraint
                new_sides[i] = min(new_sides[i], max(0.0, bound_s[i]))
                # Pairwise constraints
                for j in range(n):
                    if i == j:
                        continue
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    gap = max(dx, dy)
                    # s_i + s_j <= 2 * gap
                    # s_i <= 2*gap - s_j
                    new_sides[i] = min(new_sides[i], max(0.0, 2.0 * gap - sides[j]))
            if np.allclose(new_sides, sides, atol=1e-10):
                sides = new_sides
                break
            sides = new_sides
        return sides
    
    def penalty_and_objective(params, weight=100.0):
        """
        Objective: minimize -sum(sides) + penalty for invalid configurations.
        params: [cx_0, cy_0, cx_1, cy_1, ..., cx_9, cy_9]
        """
        cx = params[0:n]
        cy = params[n:2*n]
        
        # Check bounds
        penalty = 0.0
        for i in range(n):
            if cx[i] < -tol or cx[i] > 1.0 + tol:
                penalty += weight * (abs(cx[i] + tol) if cx[i] < -tol else abs(cx[i] - 1.0 - tol))
            if cy[i] < -tol or cy[i] > 1.0 + tol:
                penalty += weight * (abs(cy[i] + tol) if cy[i] < -tol else abs(cy[i] - 1.0 - tol))
        
        sides = compute_sides_iterative(cx, cy)
        score = -np.sum(sides) + penalty
        return score
    
    def is_valid(centers, sides):
        """Check if configuration is valid per the task contract."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        n = len(sides)
        cx = centers[:, 0]
        cy = centers[:, 1]
        # Boundary check
        for i in range(n):
            if cx[i] - sides[i]/2 < -tol or cx[i] + sides[i]/2 > 1.0 + tol:
                return False
            if cy[i] - sides[i]/2 < -tol or cy[i] + sides[i]/2 > 1.0 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True
    
    def evaluate_and_store(cx, cy):
        """Compute sides, check validity, update best if improved."""
        nonlocal best_score, best_centers, best_sides
        sides = compute_sides_iterative(cx, cy)
        centers = np.stack([cx, cy], axis=1)
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Grid (4x3, using 10 of 12 cells)
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    initial_configs.append(np.concatenate([cx, cy]))
    
    # Config 2: 5x2 grid
    k = 5
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * (1.0/2.0)  # 2 rows
    initial_configs.append(np.concatenate([cx, cy]))
    
    # Config 3: Random points
    rng = np.random.RandomState(42)
    for seed in range(20):
        rng = np.random.RandomState(seed)
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        initial_configs.append(np.concatenate([cx, cy]))
    
    # Config 4: Symmetric patterns
    # Two rows of 5
    cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
    cy = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75])
    initial_configs.append(np.concatenate([cx, cy]))
    
    # Three rows: 4, 3, 3
    cx = np.array([0.125, 0.375, 0.625, 0.875,
                   0.25, 0.5, 0.75,
                   0.25, 0.5, 0.75])
    cy = np.array([0.167, 0.167, 0.167, 0.167,
                   0.5, 0.5, 0.5,
                   0.833, 0.833, 0.833])
    initial_configs.append(np.concatenate([cx, cy]))
    
    # Evaluate initial configs
    for params in initial_configs:
        cx = params[0:n]
        cy = params[n:2*n]
        evaluate_and_store(cx, cy)
    
    # Optimize from each initial config
    bounds = [(0.0, 1.0)] * (2 * n)
    
    for config_idx, params0 in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Try L-BFGS-B
        try:
            result = minimize(
                penalty_and_objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < 0:
                cx = result.x[0:n]
                cy = result.x[n:2*n]
                evaluate_and_store(cx, cy)
        except Exception:
            pass
        
        # Also try Nelder-Mead for robustness
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                penalty_and_objective,
                params0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            cx = result.x[0:n]
            cy = result.x[n:2*n]
            evaluate_and_store(cx, cy)
        except Exception:
            pass
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
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
