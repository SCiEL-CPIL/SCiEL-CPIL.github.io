# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize

TOL = 1e-7
DEADLINE_MARGIN = 30.0

def _is_valid(centers, sides, n=10):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp_for_pattern(n, pattern):
    """
    Given a separation pattern, solve the LP to maximize sum of sides.
    Variables: s_i (side), x_i (left edge), y_i (bottom edge)
    """
    def s_idx(i): return i
    def x_idx(i): return n + i
    def y_idx(i): return 2 * n + i

    c = np.zeros(3 * n)
    for i in range(n):
        c[s_idx(i)] = -1.0

    A_ub_rows = []
    b_ub_vals = []

    # Boundary constraints
    for i in range(n):
        row = np.zeros(3 * n)
        row[s_idx(i)] = 1.0
        row[x_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

        row = np.zeros(3 * n)
        row[s_idx(i)] = 1.0
        row[y_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

    # Separation constraints
    for (i, j), axis in pattern.items():
        row = np.zeros(3 * n)
        if axis == 'x':
            # i left of j: s_i + x_i - x_j <= 0
            row[s_idx(i)] = 1.0
            row[x_idx(i)] = 1.0
            row[x_idx(j)] = -1.0
        else:
            # i below j: s_i + y_i - y_j <= 0
            row[s_idx(i)] = 1.0
            row[y_idx(i)] = 1.0
            row[y_idx(j)] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    bounds = [(0.0, 1.0)] * (3 * n)

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            s = result.x[:n]
            x = result.x[n:2*n]
            y = result.x[2*n:3*n]
            centers = np.column_stack([x + s / 2, y + s / 2])
            return centers, s, np.sum(s)
    except:
        pass
    return None, None, -np.inf

def _local_polish(centers, sides, n, max_iter=200, deadline=None):
    """Local improvement by trying to increase each side."""
    best_val = np.sum(sides)
    centers = centers.copy()
    sides = sides.copy()

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for delta in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]:
                new_sides = sides.copy()
                new_sides[i] += delta
                if new_sides[i] <= 0 or new_sides[i] > 1:
                    continue
                if _is_valid(centers, new_sides, n):
                    sides = new_sides
                    best_val = np.sum(sides)
                    improved = True
                    break
        if not improved:
            break

    return centers, sides, best_val

def _slsqp_polish(centers, sides, n, deadline=None):
    """
    NLP polish using SLSQP.
    Variables: s_i, x_i, y_i (left edge, bottom edge)
    Objective: maximize sum(s_i)
    Constraints:
      - 0 <= x_i, x_i + s_i <= 1
      - 0 <= y_i, y_i + s_i <= 1
      - Non-overlap: for each pair (i,j), at least one of:
        x_i + s_i <= x_j, x_j + s_j <= x_i, y_i + s_i <= y_j, y_j + s_j <= y_i
      We use a smooth formulation:
        max(0, |cx_i - cx_j| - (s_i + s_j)/2) <= 0 OR max(0, |cy_i - cy_j| - (s_i + s_j)/2) <= 0
      This is non-smooth, so we use the squared distance formulation:
        (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= ((s_i + s_j)/2)^2
      i.e., (cx_i - cx_j)^2 + (cy_i - cy_j)^2 - ((s_i + s_j)/2)^2 >= 0
    """
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i] = sides[i]
        x0[n + i] = centers[i, 0] - sides[i] / 2
        x0[2 * n + i] = centers[i, 1] - sides[i] / 2

    def objective(v):
        return -np.sum(v[:n])

    def objective_grad(v):
        g = np.zeros(3 * n)
        g[:n] = -1.0
        return g

    # Constraints: each returns array of values that should be >= 0
    # Type: 'ineq' means fun(x) >= 0

    constraints = []

    # Boundary: x_i >= 0, y_i >= 0
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[n + i],
            'jac': lambda v, i=i: np.array([0.0] * n + [1.0 if j == n + i else 0.0 for j in range(n)] + [0.0] * n)
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[2 * n + i],
            'jac': lambda v, i=i: np.array([0.0] * n + [0.0] * n + [1.0 if j == 2 * n + i else 0.0 for j in range(n)])
        })

    # Boundary: x_i + s_i <= 1  ->  1 - x_i - s_i >= 0
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[n + i] - v[i],
            'jac': lambda v, i=i: np.array([-1.0 if j == i else 0.0 for j in range(n)] + [-1.0 if j == n + i else 0.0 for j in range(n)] + [0.0] * n)
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[2 * n + i] - v[i],
            'jac': lambda v, i=i: np.array([-1.0 if j == i else 0.0 for j in range(n)] + [0.0] * n + [-1.0 if j == 2 * n + i else 0.0 for j in range(n)])
        })

    # Side >= 0
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i],
            'jac': lambda v, i=i: np.array([1.0 if j == i else 0.0 for j in range(n)] + [0.0] * n + [0.0] * n)
        })

    # Non-overlap: (cx_i - cx_j)^2 + (cy_i - cy_j)^2 >= ((s_i + s_j)/2)^2
    # cx_i = x_i + s_i/2, cx_j = x_j + s_j/2
    # cy_i = y_i + s_i/2, cy_j = y_j + s_j/2
    for i in range(n):
        for j in range(i + 1, n):
            def _overlap_fun(v, i=i, j=j):
                cx_i = v[n + i] + v[i] / 2.0
                cx_j = v[n + j] + v[j] / 2.0
                cy_i = v[2 * n + i] + v[i] / 2.0
                cy_j = v[2 * n + j] + v[j] / 2.0
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                r = (v[i] + v[j]) / 2.0
                return dx * dx + dy * dy - r * r

            def _overlap_jac(v, i=i, j=j):
                cx_i = v[n + i] + v[i] / 2.0
                cx_j = v[n + j] + v[j] / 2.0
                cy_i = v[2 * n + i] + v[i] / 2.0
                cy_j = v[2 * n + j] + v[j] / 2.0
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                r = (v[i] + v[j]) / 2.0

                # d/dx_i of (dx^2 + dy^2 - r^2) where dx = cx_i - cx_j
                # d(cx_i)/dx_i = 1, d(cx_i)/s_i = 0.5
                # d(dy)/dx_i = 0
                # d(r)/s_i = 0.5, d(r)/s_j = 0.5
                grad = np.zeros(3 * n)
                # w.r.t. s_i: dx * d(cx_i)/ds_i + dy * d(cy_i)/ds_i - 2*r*0.5
                # d(cx_i)/ds_i = 0.5, d(cy_i)/ds_i = 0.5
                grad[i] = dx * 0.5 + dy * 0.5 - r
                # w.r.t. s_j: dx * d(cx_j)/ds_j + dy * d(cy_j)/ds_j - 2*r*0.5
                # d(cx_j)/ds_j = 0.5, d(cy_j)/ds_j = 0.5
                grad[j] = -dx * 0.5 - dy * 0.5 - r
                # w.r.t. x_i (left edge of i): dx * 1
                grad[n + i] = dx
                # w.r.t. x_j: dx * (-1)
                grad[n + j] = -dx
                # w.r.t. y_i: dy * 1
                grad[2 * n + i] = dy
                # w.r.t. y_j: dy * (-1)
                grad[2 * n + j] = -dy
                return grad

            constraints.append({
                'type': 'ineq',
                'fun': _overlap_fun,
                'jac': _overlap_jac
            })

    bounds = [(1e-8, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n

    best_val = -np.inf
    best_centers = centers.copy()
    best_sides = sides.copy()

    # Try SLSQP from the current point
    try:
        if deadline is not None and time.time() > deadline:
            return best_centers, best_sides, best_val

        result = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-12})

        if result.success:
            s = result.x[:n]
            x = result.x[n:2 * n]
            y = result.x[2 * n:3 * n]
            c = np.column_stack([x + s / 2, y + s / 2])
            if _is_valid(c, s, n):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
    except:
        pass

    # Also try from slightly perturbed starting points
    if deadline is not None and time.time() > deadline - 5:
        return best_centers, best_sides, best_val

    rng = np.random.RandomState(123)
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        x0_perturbed = x0.copy()
        x0_perturbed[:n] *= (1.0 + 0.05 * rng.randn(n))
        x0_perturbed[n:2 * n] += 0.02 * rng.randn(n)
        x0_perturbed[2 * n:3 * n] += 0.02 * rng.randn(n)
        x0_perturbed = np.clip(x0_perturbed, 1e-8, 1.0)

        try:
            result = minimize(objective, x0_perturbed, method='SLSQP', jac=objective_grad,
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 300, 'ftol': 1e-12})
            if result.success:
                s = result.x[:n]
                x = result.x[n:2 * n]
                y = result.x[2 * n:3 * n]
                c = np.column_stack([x + s / 2, y + s / 2])
                if _is_valid(c, s, n):
                    val = np.sum(s)
                    if val > best_val:
                        best_val = val
                        best_centers = c
                        best_sides = s
        except:
            pass

    return best_centers, best_sides, best_val

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN

    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)

    # Generate structured patterns
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    patterns = []

    # All x
    patterns.append({(i, j): 'x' for i, j in pairs})
    # All y
    patterns.append({(i, j): 'y' for i, j in pairs})

    # Grid patterns
    for cols in range(1, n + 1):
        pat = {}
        for (i, j) in pairs:
            ri_i, ci_i = divmod(i, cols)
            ri_j, ci_j = divmod(j, cols)
            if ri_i == ri_j:
                pat[(i, j)] = 'x'
            else:
                pat[(i, j)] = 'y'
        patterns.append(pat)

    # Random patterns
    rng = np.random.RandomState(42)
    for _ in range(200):
        pat = {}
        for (i, j) in pairs:
            pat[(i, j)] = 'x' if rng.random() < 0.5 else 'y'
        patterns.append(pat)

    # Solve LP for each pattern
    for pat in patterns:
        if time.time() > deadline:
            break
        centers_p, sides_p, val_p = _solve_lp_for_pattern(n, pat)
        if centers_p is not None and _is_valid(centers_p, sides_p, n):
            if val_p > best_val:
                best_val = val_p
                best_centers = centers_p.copy()
                best_sides = sides_p.copy()

    # SLSQP polish from best LP solution
    if time.time() < deadline - 30:
        centers_s, sides_s, val_s = _slsqp_polish(best_centers, best_sides, n, deadline=deadline)
        if val_s > best_val:
            best_val = val_s
            best_centers = centers_s.copy()
            best_sides = sides_s.copy()

    # Local polish
    if time.time() < deadline - 10:
        centers_l, sides_l, val_l = _local_polish(best_centers, best_sides, n, max_iter=200, deadline=deadline)
        if val_l > best_val:
            best_val = val_l
            best_centers = centers_l.copy()
            best_sides = sides_l.copy()

    # Try SLSQP from a few other LP solutions
    if time.time() < deadline - 30:
        for pat in patterns[:30]:
            if time.time() > deadline:
                break
            centers_p, sides_p, val_p = _solve_lp_for_pattern(n, pat)
            if centers_p is not None and _is_valid(centers_p, sides_p, n):
                if val_p >= best_val - 0.05:
                    centers_s, sides_s, val_s = _slsqp_polish(centers_p, sides_p, n, deadline=deadline)
                    if val_s > best_val:
                        best_val = val_s
                        best_centers = centers_s.copy()
                        best_sides = sides_s.copy()

    # Final validation
    if not _is_valid(best_centers, best_sides, n):
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
