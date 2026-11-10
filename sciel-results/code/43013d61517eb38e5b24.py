# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses a smooth penalty-based optimization with multiple initializations and
    a high-precision polish pass.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 620.0  # safety margin before 630s kill

    def compute_sides(centers):
        """Compute maximum feasible side lengths given centers."""
        sides = np.ones(n, dtype=np.float64)
        # Boundary constraints
        for i in range(n):
            cx, cy = centers[i]
            max_s = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
            sides[i] = min(sides[i], max_s)
        # Pairwise non-overlap: |cx_i - cx_j| >= (s_i + s_j)/2 OR |cy_i - cy_j| >= (s_i + s_j)/2
        # For given centers, the max side s_i such that square i doesn't overlap square j
        # requires: if |cx_i-cx_j| < (s_i+s_j)/2 then we need |cy_i-cy_j| >= (s_i+s_j)/2
        # This is complex; use iterative refinement
        for _ in range(50):
            for i in range(n):
                s_i = sides[i]
                for j in range(n):
                    if i == j:
                        continue
                    s_j = sides[j]
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    # Non-overlap: dx >= (s_i + s_j)/2 OR dy >= (s_i + s_j)/2
                    # If currently overlapping, reduce s_i
                    half_sum = (s_i + s_j) / 2.0
                    if dx < half_sum and dy < half_sum:
                        # Need to reduce so that either dx >= (s_i+s_j)/2 or dy >= (s_i+s_j)/2
                        # Reduce s_i: new constraint dx >= (s_i' + s_j)/2 => s_i' <= 2*dx - s_j
                        # Or dy >= (s_i' + s_j)/2 => s_i' <= 2*dy - s_j
                        max_allowed = max(2.0 * dx - s_j, 2.0 * dy - s_j)
                        if max_allowed < s_i:
                            sides[i] = max(max_allowed, 1e-12)
            # Also re-check boundaries
            for i in range(n):
                cx, cy = centers[i]
                max_s = min(cx, 1.0 - cx, cy, 1.0 - cy) * 2.0
                if sides[i] > max_s:
                    sides[i] = max(max_s, 1e-12)
        return sides

    def objective(params):
        """Negative sum of sides (to minimize)."""
        centers = params[:2 * n].reshape(n, 2)
        sides = compute_sides(centers)
        return -np.sum(sides)

    def make_initialization(seed):
        """Create a random initialization."""
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(n, 2))
        return centers.flatten()

    best_params = None
    best_val = -np.inf

    # Grid-based initializations
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    grid_centers = np.stack([cx, cy], axis=1)
    best_params = grid_centers.flatten().copy()
    best_val = -np.sum(compute_sides(grid_centers))

    # Try multiple random initializations with L-BFGS-B
    n_attempts = 0
    max_attempts = 30
    
    for seed in range(100):
        if time.time() > deadline:
            break
        if n_attempts >= max_attempts:
            break
        
        x0 = make_initialization(seed)
        # Add small perturbation to break symmetry
        x0 += np.random.RandomState(seed * 1000 + 42).uniform(-0.05, 0.05, size=x0.shape)
        # Clip to valid range
        x0[:n] = np.clip(x0[:n], 0.0, 1.0)
        x0[n:] = np.clip(x0[n:], 0.0, 1.0)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          bounds=[(0, 1)] * (2 * n),
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            n_attempts += 1
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass

    # Also try the grid initialization with optimization
    try:
        x0 = grid_centers.flatten().copy()
        res = minimize(objective, x0, method='L-BFGS-B',
                      bounds=[(0, 1)] * (2 * n),
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        n_attempts += 1
        val = res.fun
        if val < best_val:
            best_val = val
            best_params = res.x.copy()
    except Exception:
        pass

    centers = best_params[:2 * n].reshape(n, 2)
    sides = compute_sides(centers)

    # High-precision polish: use bisection to tighten side lengths
    for _ in range(80):
        if time.time() > deadline:
            break
        for i in range(n):
            # Try to increase side i slightly
            lo, hi = sides[i], min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]) * 2.0
            if hi <= lo:
                continue
            for _ in range(30):
                mid = (lo + hi) / 2.0
                # Check if increasing side i to mid causes overlap
                feasible = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    half_sum = (mid + sides[j]) / 2.0
                    if dx < half_sum - 1e-12 and dy < half_sum - 1e-12:
                        feasible = False
                        break
                if feasible:
                    lo = mid
                else:
                    hi = mid
            sides[i] = lo

    # Final validation and cleanup
    centers = np.clip(centers, 0, 1)
    sides = np.clip(sides, 1e-12, None)
    # Ensure sides are consistent with centers
    for i in range(n):
        max_s = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]) * 2.0
        sides[i] = min(sides[i], max_s)

    return centers.astype(np.float64), sides.astype(np.float64)


# EVOLVE-BLOCK-END
