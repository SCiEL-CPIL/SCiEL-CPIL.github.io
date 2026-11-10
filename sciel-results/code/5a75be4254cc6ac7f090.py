# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 10
TOL = 1e-7

def _build_grid4():
    side = 0.25
    centers = np.array([
        [0.125, 0.125],
        [0.375, 0.125],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.125, 0.375],
        [0.375, 0.375],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.125, 0.625],
        [0.375, 0.625],
    ])
    sides = np.full(10, side)
    return centers, sides

def _build_two_halves():
    # Left half: one 0.5 square + four 0.25 squares; same for right half
    centers = np.array([
        [0.25, 0.25],   # L-Big
        [0.125, 0.625],
        [0.375, 0.625],
        [0.125, 0.875],
        [0.375, 0.875],
        [0.75, 0.25],   # R-Big
        [0.625, 0.625],
        [0.875, 0.625],
        [0.625, 0.875],
        [0.875, 0.875],
    ])
    sides = np.array([0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25])
    return centers, sides

def _validity_penalty(centers, sides):
    """
    Large penalty if any constraint is violated.
    Uses soft constraints (smooth) for optimization.
    """
    cx = centers[:, 0]
    cy = centers[:, 1]
    s = sides

    pen = 0.0

    # Inside the unit square (with tolerance)
    lo = np.minimum(
        np.minimum(cx - 0.5 * s, cy - 0.5 * s),
        np.minimum(1.0 - (cx + 0.5 * s), 1.0 - (cy + 0.5 * s))
    )
    pen += np.sum(np.maximum(0.0, TOL - lo) ** 2) * 1e4

    # Pairwise non-overlap:
    # Overlap forbidden if |cx_i - cx_j| < t and |cy_i - cy_j| < t, t = (s_i+s_j)/2 - tol
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            t = 0.5 * (s[i] + s[j]) - TOL
            # Want: dx >= t OR dy >= t
            # Penalty when both are violated:
            # Use a smooth hinge: max(0, t - dx) * max(0, t - dy)
            a = max(0.0, t - dx)
            b = max(0.0, t - dy)
            pen += a * b * 1e4

    return pen

def _objective(x):
    """
    x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    """
    centers = x[0:3 * N].reshape(N, 3)
    cx = centers[:, 0]
    cy = centers[:, 1]
    s = centers[:, 2]

    # Keep sides positive
    s = np.maximum(s, 1e-6)

    # Sum of sides to maximize -> minimize negative
    obj = -np.sum(s)

    # Add penalty
    obj += _validity_penalty(
        np.column_stack([cx, cy]), s
    )

    return obj

def _pack_from_seed(centers0, sides0, deadline):
    """
    Local search starting from a given feasible packing.
    """
    x0 = np.empty(3 * N)
    x0[0:3 * N:3] = centers0[:, 0]
    x0[1:3 * N:3] = centers0[:, 1]
    x0[2:3 * N:3] = sides0

    best_val = -np.sum(sides0)
    best_x = x0.copy()

    n_iter = 0
    while time.time() < deadline:
        try:
            res = minimize(
                _objective, x0, method='L-BFGS-B',
                options={'maxiter': 200, 'maxls': 20, 'gtol': 1e-10, 'disp': False}
            )
        except Exception:
            break

        n_iter += 1

        # Evaluate the result
        cx = res.x[0:3 * N:3]
        cy = res.x[1:3 * N:3]
        s = np.maximum(res.x[2:3 * N:3], 1e-6)

        val = -np.sum(s) - _validity_penalty(np.column_stack([cx, cy]), s)
        # Actually objective returns -sum(s) + penalty. We want to maximize sum(s) - penalty.
        # Let's just compute directly:
        raw = -res.fun  # This is sum(s) - penalty (approximately, since fun = -sum(s) + pen)
        # But _objective returns -sum(s) + pen. So -res.fun = sum(s) - pen.
        # If pen is 0, raw = sum(s).

        # Check if valid enough (small penalty)
        c = np.column_stack([cx, cy])
        pen = _validity_penalty(c, s)
        if pen < 1e-4:
            if raw > best_val:
                best_val = raw
                best_x = res.x.copy()
                x0 = res.x.copy()
                # If improvement is small, we may be converging
                if raw - (best_val - raw) < 1e-6:  # This logic is a bit off; let's just check improvement
                    pass
        else:
            # If invalid, don't update best, but maybe restart from a perturbed x0
            pass

        # Check convergence: if no significant improvement in last few iters, stop
        if n_iter > 5 and best_val > -np.sum(sides0) + 1e-4:
            # We improved significantly; maybe continue a bit more
            pass

        # Simple stopping: if the last optimization didn't improve much, stop
        last_improve = raw - (best_val - raw) if raw > best_val - raw else 0
        # Actually let's track improvement differently
        if n_iter > 3:
            prev_best = best_val
            # Re-evaluate: if best_x hasn't changed in value much, stop
            # For simplicity, stop after 20 iterations or if improvement < 1e-6
            if raw - prev_best < 1e-6:
                break

    # Extract best
    cx = best_x[0:3 * N:3]
    cy = best_x[1:3 * N:3]
    s = np.maximum(best_x[2:3 * N:3], 1e-6)

    # Final clamping to ensure validity
    for i in range(N):
        # Clamp side
        s[i] = min(s[i], 1.0)
        # Clamp center so square is inside [0,1]
        cx[i] = np.clip(cx[i], s[i] / 2, 1.0 - s[i] / 2)
        cy[i] = np.clip(cy[i], s[i] / 2, 1.0 - s[i] / 2)

    centers = np.column_stack([cx, cy])

    # Verify validity; if not, fall back to seed
    pen = _validity_penalty(centers, s)
    if pen > 1e-4:
        return centers0, sides0

    return centers, s

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + 600 - 10  # 10s safety margin

    # Initial packings
    grid_centers, grid_sides = _build_grid4()
    half_centers, half_sides = _build_two_halves()

    best_centers = grid_centers
    best_sides = grid_sides
    best_val = np.sum(grid_sides)

    # Try local search from grid
    c1, s1 = _pack_from_seed(grid_centers, grid_sides, deadline)
    val1 = np.sum(s1)
    if val1 > best_val:
        best_val = val1
        best_centers = c1
        best_sides = s1

    # Try local search from two-halves
    c2, s2 = _pack_from_seed(half_centers, half_sides, deadline)
    val2 = np.sum(s2)
    if val2 > best_val:
        best_val = val2
        best_centers = c2
        best_sides = s2

    # Final validity check
    # Ensure all sides > 0 and finite
    best_sides = np.where(best_sides > 0, best_sides, 1e-9)
    best_centers = np.where(np.isfinite(best_centers), best_centers, 0.5)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
