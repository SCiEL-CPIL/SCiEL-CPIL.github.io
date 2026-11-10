# EVOLVE-BLOCK-START
import time
import numpy as np


def _validate(centers, sides, n=10, tol=1e-7):
    """Return True if the configuration is valid (inside box, no overlaps, finite, positive)."""
    if centers is None or sides is None:
        return False
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    if np.any(sides <= 0):
        return False
    # Inside the unit square (with tolerance)
    if np.any(centers[0] - sides / 2 < -tol) or np.any(centers[0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[1] - sides / 2 < -tol) or np.any(centers[1] + sides / 2 > 1 + tol):
        return False
    # Pairwise non-overlap: overlap if |dx| < t and |dy| < t where t = (si+sj)/2 - tol
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(x, n=10, alpha=1e4, beta=1e4):
    """
    Objective = -sum(sides) + overlap_penalty + boundary_penalty.
    x layout: [s0..s9, cx0..cx9, cy0..cy9] (length 30).
    Lower is better.
    """
    s = x[:n]
    cx = x[n:2 * n]
    cy = x[2 * n:3 * n]

    obj = -np.sum(s)

    # Boundary penalties: push centers so squares stay inside [0,1]^2
    # For each square: cx must be in [s/2, 1 - s/2], similarly cy.
    lo_x = s / 2
    hi_x = 1 - s / 2
    lo_y = s / 2
    hi_y = 1 - s / 2

    # Soft violation: amount by which cx < lo_x or cx > hi_x
    viol_x = np.maximum(lo_x - cx, 0) + np.maximum(cx - hi_x, 0)
    viol_y = np.maximum(lo_y - cy, 0) + np.maximum(cy - hi_y, 0)
    obj += alpha * np.sum(viol_x ** 2) + alpha * np.sum(viol_y ** 2)

    # Pairwise overlap penalties (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            t = (s[i] + s[j]) / 2
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            # Overlap depth in x: max(0, t - dx); in y: max(0, t - dy)
            # If both positive, they overlap. Use product as a smooth penalty.
            ox = max(0.0, t - dx)
            oy = max(0.0, t - dy)
            if ox > 0 and oy > 0:
                obj += beta * ox * oy

    return obj


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares packed in the unit square.
    Uses a deterministic multi-start local optimization with a smooth repulsion model.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s kill

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def try_config(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _validate(centers, sides, n):
            sc = float(np.sum(sides))
            if sc > best_score:
                best_score = sc
                best_centers = centers.copy()
                best_sides = sides.copy()

    # --- Initial seed: simple grid (always valid) ---
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    centers0 = np.stack([cx0, cy0], axis=1).astype(np.float64)
    sides0 = np.full(n, side, dtype=np.float64)
    try_config(centers0, sides0)

    # --- Optimization helper using scipy ---
    from scipy.optimize import minimize

    def optimize_from(x0, max_iter=2000, tol_opt=1e-10):
        """Run L-BFGS-B from x0, return (centers, sides, valid, score)."""
        # Variable bounds:
        # s_i in (0, 1), cx_i in (0, 1), cy_i in (0, 1)
        lb = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n)])
        ub = np.concatenate([np.ones(n), np.ones(n), np.ones(n)])
        # Tighter lower bound for sides to avoid zero
        lb[:n] = 1e-6

        res = minimize(
            _objective,
            x0,
            args=(n,),
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': max_iter, 'ftol': tol_opt, 'gtol': 1e-8},
        )

        x_opt = res.x
        s = x_opt[:n]
        cx = x_opt[n:2 * n]
        cy = x_opt[2 * n:3 * n]
        centers = np.stack([cx, cy], axis=1)
        sides = s

        # Post-process: clamp to valid range and check
        # Ensure sides > 0
        sides = np.maximum(sides, 1e-6)
        # Clamp centers so squares stay inside [0,1]^2
        cx_clamped = np.clip(cx, sides / 2, 1 - sides / 2)
        cy_clamped = np.clip(cy, sides / 2, 1 - sides / 2)
        centers_clamped = np.stack([cx_clamped, cy_clamped], axis=1)

        return centers_clamped, sides

    # --- Multi-start optimization ---
    np.random.seed(42)

    # Start 1: from grid
    x0 = np.concatenate([sides0, centers0[:, 0], centers0[:, 1]])
    c, s = optimize_from(x0, max_iter=3000)
    try_config(c, s)

    # If deadline approaching, stop
    if time.time() > deadline - 60:
        pass
    else:
        # Start 2: slightly perturbed grid
        for trial in range(5):
            if time.time() > deadline - 60:
                break
            # Random perturbation of grid positions and sides
            pert = np.random.normal(0, 0.05, size=(n, 2))
            cx_p = np.clip(centers0[:, 0] + pert[:, 0], sides0 / 2, 1 - sides0 / 2)
            cy_p = np.clip(centers0[:, 1] + pert[:, 1], sides0 / 2, 1 - sides0 / 2)
            s_p = np.clip(sides0 + np.random.normal(0, 0.02, n), 0.1, 0.4)
            c_p = np.stack([cx_p, cy_p], axis=1)
            x0 = np.concatenate([s_p, cx_p, cy_p])
            c, s = optimize_from(x0, max_iter=2000)
            try_config(c, s)

    # Start 3: random feasible initialization
    if time.time() > deadline - 60:
        pass
    else:
        for trial in range(10):
            if time.time() > deadline - 60:
                break
            # Generate random non-overlapping configuration
            # Place squares one by one with random sizes
            cx_r = np.zeros(n)
            cy_r = np.zeros(n)
            s_r = np.zeros(n)
            # Random sizes between 0.1 and 0.35
            s_r = np.random.uniform(0.1, 0.35, n)
            # Sort by size descending for placement
            order = np.argsort(-s_r)
            for i in order:
                si = s_r[i]
                # Random position
                for attempt in range(100):
                    cx_try = np.random.uniform(si / 2, 1 - si / 2)
                    cy_try = np.random.uniform(si / 2, 1 - si / 2)
                    # Check against already placed
                    ok = True
                    for j in order:
                        if s_r[j] == 0:
                            continue
                        tj = (si + s_r[j]) / 2
                        if abs(cx_try - cx_r[j]) < tj and abs(cy_try - cy_r[j]) < tj:
                            ok = False
                            break
                    if ok:
                        cx_r[i] = cx_try
                        cy_r[i] = cy_try
                        break
                else:
                    # Fallback: place at corner
                    cx_r[i] = si / 2
                    cy_r[i] = si / 2

            c_r = np.stack([cx_r, cy_r], axis=1)
            x0 = np.concatenate([s_r, cx_r, cy_r])
            c, s = optimize_from(x0, max_iter=1500)
            try_config(c, s)

    # Start 4: structured asymmetric layouts
    if time.time() > deadline - 60:
        pass
    else:
        # Layout: 2 large squares + 8 small in remaining space
        # Try: two 0.4x0.4 squares in bottom, rest above
        s_struct = np.array([0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
        cx_struct = np.array([0.2, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8])
        cy_struct = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8])
        c_struct = np.stack([cx_struct, cy_struct], axis=1)
        x0 = np.concatenate([s_struct, cx_struct, cy_struct])
        c, s = optimize_from(x0, max_iter=2000)
        try_config(c, s)

        # Layout: one big + nine small
        s_struct2 = np.array([0.5, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        cx_struct2 = np.array([0.25, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        cy_struct2 = np.array([0.25, 0.075, 0.225, 0.375, 0.525, 0.675, 0.825, 0.075, 0.225, 0.375])
        c_struct2 = np.stack([cx_struct2, cy_struct2], axis=1)
        x0 = np.concatenate([s_struct2, cx_struct2, cy_struct2])
        c, s = optimize_from(x0, max_iter=2000)
        try_config(c, s)

    # Final safety: if no valid config found, return grid
    if best_centers is None:
        return centers0, sides0

    return best_centers, best_sides


# EVOLVE-BLOCK-END
