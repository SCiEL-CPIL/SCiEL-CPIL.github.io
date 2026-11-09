# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7
_DEADLINE = 630.0


def _validate(centers, sides, n=10):
    """Check all constraints. Return (is_valid, value)."""
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False, 0.0
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False, 0.0
    if np.any(sides <= 0):
        return False, 0.0
    for i in range(n):
        if centers[i, 0] - sides[i] / 2 < -_TOL or centers[i, 0] + sides[i] / 2 > 1 + _TOL:
            return False, 0.0
        if centers[i, 1] - sides[i] / 2 < -_TOL or centers[i, 1] + sides[i] / 2 > 1 + _TOL:
            return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False, 0.0
    value = float(np.sum(sides))
    return True, value


def _objective(x, n=10):
    """Minimize negative sum of side lengths. x = [cx, cy, s] interleaved."""
    sides = x[2 * np.arange(n): 3 * n]  # s_i at indices 2i+2
    return -np.sum(sides)


def _constraints(n=10):
    """Build constraint functions for SLSQP."""
    cons = []
    # Boundary constraints: cx_i - s_i/2 >= 0  =>  cx_i - s_i/2 >= 0
    # cx_i + s_i/2 <= 1  =>  1 - cx_i - s_i/2 >= 0
    # Same for cy
    for i in range(n):
        ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
        # cx_i - s_i/2 >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda x, ci=ci, si=si: x[ci] - x[si] / 2.0,
                     'jac': lambda x, ci=ci, si=si: _grad_boundary(x, n, i, 'left')})
        # 1 - cx_i - s_i/2 >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda x, ci=ci, si=si: 1.0 - x[ci] - x[si] / 2.0,
                     'jac': lambda x, ci=ci, si=si: _grad_boundary(x, n, i, 'right')})
        # cy_i - s_i/2 >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda x, cji=cji, si=si: x[cji] - x[si] / 2.0,
                     'jac': lambda x, cji=cji, si=si: _grad_boundary(x, n, i, 'bottom')})
        # 1 - cy_i - s_i/2 >= 0
        cons.append({'type': 'ineq',
                     'fun': lambda x, cji=cji, si=si: 1.0 - x[cji] - x[si] / 2.0,
                     'jac': lambda x, cji=cji, si=si: _grad_boundary(x, n, i, 'top')})

    # Non-overlap: for each pair (i,j), require
    #   |cx_i - cx_j| >= (s_i + s_j)/2  OR  |cy_i - cy_j| >= (s_i + s_j)/2
    # This is a disjunction. We handle it by adding BOTH constraints as
    # "soft" and relying on the optimizer to find a feasible point where
    # at least one holds. Actually, SLSQP needs individual constraints.
    # Better approach: use the L-infinity trick.
    # The squares don't overlap iff:
    #   max(|cx_i - cx_j|, |cy_i - cy_j|) >= (s_i + s_j)/2
    # Equivalently, we need:
    #   |cx_i - cx_j| - (s_i+s_j)/2 >= 0  OR  |cy_i - cy_j| - (s_i+s_j)/2 >= 0
    # This is non-smooth and disjunctive.
    #
    # Alternative smooth formulation:
    # Use the fact that squares are axis-aligned. The non-overlap condition
    # can be written as:
    #   (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= ((s_i+s_j)/2)^2  ... NO, that's for circles.
    #
    # For axis-aligned squares, the correct condition is:
    #   |cx_i - cx_j| >= (s_i+s_j)/2  OR  |cy_i - cy_j| >= (s_i+s_j)/2
    #
    # A smooth surrogate: penalize the minimum of the two separations.
    # min(|dx| - (si+sj)/2, |dy| - (si+sj)/2) >= 0
    # This is equivalent to the OR condition.
    #
    # We can write this as a single smooth constraint using:
    #   f = min(a, b) where a = |dx| - (si+sj)/2, b = |dy| - (si+sj)/2
    # min(a,b) = (a+b - |a-b|)/2
    # So the constraint is: (a+b - |a-b|)/2 >= 0
    # This is smooth except where a=b, but SLSQP can handle it numerically.
    
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(_make_nonoverlap_constraint(i, j, n))

    return cons


def _grad_boundary(x, n, i, which):
    """Jacobian for boundary constraints."""
    grad = np.zeros(3 * n)
    ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
    if which == 'left':
        grad[ci] = 1.0
        grad[si] = -0.5
    elif which == 'right':
        grad[ci] = -1.0
        grad[si] = -0.5
    elif which == 'bottom':
        grad[cji] = 1.0
        grad[si] = -0.5
    elif which == 'top':
        grad[cji] = -1.0
        grad[si] = -0.5
    return grad


def _make_nonoverlap_constraint(i, j, n):
    """Create a smooth constraint for non-overlap of squares i and j."""
    ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
    cj, cjj, sj = 2 * j, 2 * j + 1, 2 * j + 2

    def fun(x):
        dx = x[ci] - x[cj]
        dy = x[cji] - x[cjj]
        a = abs(dx) - (x[si] + x[sj]) / 2.0
        b = abs(dy) - (x[si] + x[sj]) / 2.0
        # min(a, b) = (a + b - |a - b|) / 2
        return (a + b - abs(a - b)) / 2.0

    def jac(x):
        dx = x[ci] - x[cj]
        dy = x[cji] - x[cjj]
        a = abs(dx) - (x[si] + x[sj]) / 2.0
        b = abs(dy) - (x[si] + x[sj]) / 2.0
        grad = np.zeros(3 * n)
        
        # Gradient of min(a,b) = (a+b-|a-b|)/2
        # d/da: 1/2 * (1 - sign(a-b))
        # d/db: 1/2 * (1 + sign(a-b))
        diff = a - b
        if abs(diff) < 1e-12:
            # At the kink, use average gradient
            da_da = 0.5
            db_da = 0.5
        else:
            sgn = 1.0 if diff > 0 else -1.0
            da_da = 0.5 * (1.0 - sgn)  # 0 if a>b, 1 if a<b
            db_da = 0.5 * (1.0 + sgn)  # 1 if a>b, 0 if a<b

        # da/dx:
        # a = |dx| - (si+sj)/2
        # d|dx|/d(ci) = sign(dx), d|dx|/d(cj) = -sign(dx)
        # d(si+sj)/2/d(si) = 0.5, d(si+sj)/2/d(sj) = 0.5
        sgn_dx = 1.0 if dx >= 0 else -1.0
        sgn_dy = 1.0 if dy >= 0 else -1.0

        grad[ci] = da_da * sgn_dx
        grad[cj] = da_da * (-sgn_dx)
        grad[si] = da_da * (-0.5)
        grad[sj] = da_da * (-0.5)

        # db/dx:
        # b = |dy| - (si+sj)/2
        grad[cji] = db_da * sgn_dy
        grad[cjj] = db_da * (-sgn_dy)
        grad[si] += db_da * (-0.5)
        grad[sj] += db_da * (-0.5)

        return grad

    return {'type': 'ineq', 'fun': fun, 'jac': jac}


def _solve_nlp(x0, n=10, maxiter=2000):
    """Solve the NLP from initial guess x0. Return (value, centers, sides) or None."""
    cons = _constraints(n)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((1e-9, 1.0))  # s

    try:
        res = minimize(_objective, x0, args=(n,), method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-12})
    except Exception:
        return None

    if not res.success:
        # Still check if the result is feasible
        pass

    x = res.x
    centers = np.column_stack([x[2 * np.arange(n)], x[2 * np.arange(n) + 1]]).astype(np.float64)
    sides = x[2 * np.arange(n) + 2].astype(np.float64)

    valid, value = _validate(centers, sides, n)
    if not valid:
        return None
    return value, centers, sides


def _generate_initial_guesses(n=10):
    """Generate diverse initial guesses for the NLP solver."""
    guesses = []

    # 1. Grid layouts
    for rows in [2, 3, 4, 5]:
        for cols in [2, 3, 4, 5]:
            if rows * cols >= n:
                side = min(1.0 / rows, 1.0 / cols)
                cx = np.zeros(n)
                cy = np.zeros(n)
                s = np.full(n, side)
                for i in range(n):
                    r = i // cols
                    c = i % cols
                    cx[i] = (c + 0.5) * (1.0 / cols)
                    cy[i] = (r + 0.5) * (1.0 / rows)
                x0 = np.zeros(3 * n)
                x0[2 * np.arange(n)] = cx
                x0[2 * np.arange(n) + 1] = cy
                x0[2 * np.arange(n) + 2] = s
                guesses.append(x0)

    # 2. 3x3 grid + 1 in corner (known good structure)
    # 9 squares in 3x3 grid of size 1/3, 10th smaller
    side_main = 1.0 / 3.0
    cx = np.zeros(n)
    cy = np.zeros(n)
    s = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        cx[i] = (c + 0.5) * side_main
        cy[i] = (r + 0.5) * side_main
        s[i] = side_main
    # 10th square: place in a corner with small size
    cx[9] = 0.15
    cy[9] = 0.15
    s[9] = 0.3
    x0 = np.zeros(3 * n)
    x0[2 * np.arange(n)] = cx
    x0[2 * np.arange(n) + 1] = cy
    x0[2 * np.arange(n) + 2] = s
    guesses.append(x0)

    # 3. Two-row layout: 5 squares per row
    for row_split in [0.4, 0.5, 0.6]:
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        s[:5] = 0.2
        s[5:] = 0.2
        for i in range(5):
            cx[i] = (i + 0.5) * 0.2
            cy[i] = row_split / 2
        for i in range(5):
            cx[5 + i] = (i + 0.5) * 0.2
            cy[5 + i] = row_split + (1 - row_split) / 2
        x0 = np.zeros(3 * n)
        x0[2 * np.arange(n)] = cx
        x0[2 * np.arange(n) + 1] = cy
        x0[2 * np.arange(n) + 2] = s
        guesses.append(x0)

    # 4. Random perturbations of grid
    rng = np.random.RandomState(42)
    for _ in range(5):
        rows, cols = 3, 4
        side = min(1.0 / rows, 1.0 / cols)
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.full(n, side)
        for i in range(n):
            r = i // cols
            c = i % cols
            cx[i] = (c + 0.5) * (1.0 / cols) + rng.uniform(-0.02, 0.02)
            cy[i] = (r + 0.5) * (1.0 / rows) + rng.uniform(-0.02, 0.02)
        # Clip to valid range
        for i in range(n):
            cx[i] = np.clip(cx[i], s[i] / 2, 1 - s[i] / 2)
            cy[i] = np.clip(cy[i], s[i] / 2, 1 - s[i] / 2)
        x0 = np.zeros(3 * n)
        x0[2 * np.arange(n)] = cx
        x0[2 * np.arange(n) + 1] = cy
        x0[2 * np.arange(n) + 2] = s
        guesses.append(x0)

    return guesses


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    deadline = t0 + _DEADLINE - 30.0

    best_value = 0.0
    best_centers = None
    best_sides = None

    # Generate initial guesses
    guesses = _generate_initial_guesses(n)

    # Solve NLP from each initial guess
    for idx, x0 in enumerate(guesses):
        if time.time() > deadline:
            break

        # Limit iterations based on remaining time
        remaining = deadline - time.time()
        maxiter = min(2000, max(500, int(remaining * 10)))

        res = _solve_nlp(x0, n, maxiter=maxiter)
        if res is not None:
            val, centers, sides = res
            if val > best_value:
                best_value = val
                best_centers = centers.copy()
                best_sides = sides.copy()

    # If we have a good result, try to improve with more iterations from the best
    if best_centers is not None and time.time() < deadline - 60:
        x0_best = np.zeros(3 * n)
        x0_best[2 * np.arange(n)] = best_centers[:, 0]
        x0_best[2 * np.arange(n) + 1] = best_centers[:, 1]
        x0_best[2 * np.arange(n) + 2] = best_sides

        remaining = deadline - time.time()
        maxiter = min(5000, max(1000, int(remaining * 5)))
        res = _solve_nlp(x0_best, n, maxiter=maxiter)
        if res is not None:
            val, centers, sides = res
            if val > best_value:
                best_value = val
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Fallback: simple grid
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx_arr = np.arange(n)
        cols = idx_arr % k
        rows = idx_arr // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    # Final validation
    valid, _ = _validate(best_centers, best_sides, n)
    if not valid:
        # Emergency fallback
        k = 4
        side = 1.0 / k
        idx_arr = np.arange(n)
        cols = idx_arr % k
        rows = idx_arr // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides


# EVOLVE-BLOCK-END
