"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _softplus(x, beta=10.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(-beta * x)) / beta)


def _objective_and_grad(params, beta=10.0):
    """
    params: array of shape (3*N,) = [x0,y0,r0, x1,y1,r1, ...]
    Objective: -sum(r) + penalty for overlaps and boundary violations
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    # Sum of radii (to maximize)
    sum_r = np.sum(r)

    # Boundary penalties: circles must stay inside [0,1]^2
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0, same for y
    # Penalty if violated
    bound_viol = (
        _softplus(-(x - r), beta) +
        _softplus(-(1.0 - x - r), beta) +
        _softplus(-(y - r), beta) +
        _softplus(-(1.0 - y - r), beta)
    )

    # Overlap penalties: for each pair (i,j), require dist(i,j) >= r_i + r_j
    # i.e., sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    overlap_penalty = 0.0
    overlap_grad_x = np.zeros(n)
    overlap_grad_y = np.zeros(n)
    overlap_grad_r = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-15)
            gap = dist - r[i] - r[j]  # should be >= 0
            penalty = _softplus(-gap, beta)
            overlap_penalty += penalty

            # Gradient of softplus(-gap) w.r.t. gap = -sigmoid(beta*gap) approx
            # d(softplus(-gap))/d(gap) = -sigmoid(beta * gap) for the smooth part
            # Actually: softplus(-gap) = log(1+exp(-beta*(-gap)))/beta = log(1+exp(beta*gap))/beta
            # d/d(gap) = beta*exp(beta*gap)/(beta*(1+exp(beta*gap))) = sigmoid(beta*gap)
            # So d(penalty)/d(gap) = sigmoid(beta*gap)
            # And d(gap)/d(x_i) = dx/dist, d(gap)/d(x_j) = -dx/dist
            # d(gap)/d(r_i) = -1, d(gap)/d(r_j) = -1

            sig = 1.0 / (1.0 + np.exp(-beta * gap))
            # gradient contribution to penalty
            gx_i = sig * dx / dist
            gx_j = -sig * dx / dist
            gy_i = sig * dy / dist
            gy_j = -sig * dy / dist
            gr_i = -sig
            gr_j = -sig

            overlap_grad_x[i] += gx_i
            overlap_grad_x[j] += gx_j
            overlap_grad_y[i] += gy_i
            overlap_grad_y[j] += gy_j
            overlap_grad_r[i] += gr_i
            overlap_grad_r[j] += gr_j

    # Boundary gradients
    # d(softplus(-(x-r)))/dx = -sigmoid(beta*(x-r)) ... wait let me redo
    # f = softplus(-(x-r)) = log(1+exp(beta*(x-r)))/beta
    # df/dx = sigmoid(beta*(x-r))
    # df/dr = -sigmoid(beta*(x-r))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    grad_x = -np.ones(n)  # from -sum(r)... no, -sum(r) gives grad 0 for x
    # Actually objective = -sum_r + overlap_penalty + bound_penalty
    # d(-sum_r)/dx = 0, d(-sum_r)/dr = -1

    # Boundary gradient contributions
    # For x: softplus(-(x-r)) + softplus(-(1-x-r))
    # d/dx: sigmoid(beta*(x-r)) - sigmoid(beta*(1-x-r))
    # d/dr: -sigmoid(beta*(x-r)) - sigmoid(beta*(1-x-r))
    bx = sigmoid(beta * (x - r)) - sigmoid(beta * (1.0 - x - r))
    by = sigmoid(beta * (y - r)) - sigmoid(beta * (1.0 - y - r))
    br = -sigmoid(beta * (x - r)) - sigmoid(beta * (1.0 - x - r)) - sigmoid(beta * (y - r)) - sigmoid(beta * (1.0 - y - r))

    # Total gradient
    grad_x_total = overlap_grad_x + bx
    grad_y_total = overlap_grad_y + by
    grad_r_total = -np.ones(n) + overlap_grad_r + br

    obj = -sum_r + overlap_penalty + np.sum(bound_viol)

    grad = np.empty(3 * n)
    grad[0::3] = grad_x_total
    grad[1::3] = grad_y_total
    grad[2::3] = grad_r_total

    return obj, grad


def _is_valid(circles, tol=1e-9):
    """Check if circles are valid (inside unit square, no overlaps)."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Boundary check
    if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol):
        return False

    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _repair(circles):
    """Ensure circles are strictly inside and non-overlapping by slight adjustments."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    # Clamp radii to be positive
    r = np.maximum(r, 1e-6)

    # Adjust positions to satisfy boundary
    for i in range(N):
        r[i] = min(r[i], 0.5 - 1e-9)
        x[i] = np.clip(x[i], r[i] + 1e-9, 1.0 - r[i] - 1e-9)
        y[i] = np.clip(y[i], r[i] + 1e-9, 1.0 - r[i] - 1e-9)

    # Iteratively separate overlapping circles
    for _ in range(200):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-15)
                min_dist = r[i] + r[j] + 1e-9
                if dist < min_dist:
                    # Push apart
                    overlap = min_dist - dist
                    if dist < 1e-12:
                        # Random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    # Move both circles apart
                    x[i] += dx * overlap * 0.5
                    y[i] += dy * overlap * 0.5
                    x[j] -= dx * overlap * 0.5
                    y[j] -= dy * overlap * 0.5
                    moved = True
        if not moved:
            break

    # Final clamp
    for i in range(N):
        r[i] = min(r[i], 0.5 - 1e-9)
        x[i] = np.clip(x[i], r[i] + 1e-9, 1.0 - r[i] - 1e-9)
        y[i] = np.clip(y[i], r[i] + 1e-9, 1.0 - r[i] - 1e-9)

    circles_repaired = np.column_stack([x, y, r])
    return circles_repaired


def _score(circles):
    """Compute the sum of radii."""
    return np.sum(circles[:, 2])


def construct():
    start_time = time.time()
    best_circles = None
    best_score = -np.inf

    # Initial grid solution as baseline
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    baseline = np.array(cells[:N])
    if _is_valid(baseline):
        best_circles = baseline.copy()
        best_score = _score(baseline)

    # Multi-start optimization with softplus penalty
    n_starts = 30
    rng = np.random.default_rng(42)

    for start in range(n_starts):
        if time.time() - start_time > DEADLINE:
            break

        # Initialize parameters
        if start == 0:
            # Start from grid
            x0 = baseline[:, 0].copy()
            y0 = baseline[:, 1].copy()
            r0 = baseline[:, 2].copy()
        else:
            # Random initialization
            x0 = rng.uniform(0.1, 0.9, N)
            y0 = rng.uniform(0.1, 0.9, N)
            r0 = rng.uniform(0.02, 0.12, N)

        params0 = np.empty(3 * N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0

        # Bounds: x,y in [0,1], r in [0, 0.5]
        lb = np.zeros(3 * N)
        ub = np.zeros(3 * N)
        ub[0::3] = 1.0
        ub[1::3] = 1.0
        ub[2::3] = 0.5

        # Run optimization
        try:
            result = minimize(
                _objective_and_grad,
                params0,
                jac=True,
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
        except Exception:
            continue

        # Extract solution
        x_opt = result.x[0::3]
        y_opt = result.x[1::3]
        r_opt = result.x[2::3]
        circles_opt = np.column_stack([x_opt, y_opt, r_opt])

        # Repair
        circles_repaired = _repair(circles_opt)

        # Check validity
        if _is_valid(circles_repaired):
            sc = _score(circles_repaired)
            if sc > best_score:
                best_score = sc
                best_circles = circles_repaired.copy()

        # Also try the un-repaired version if valid
        if _is_valid(circles_opt):
            sc = _score(circles_opt)
            if sc > best_score:
                best_score = sc
                best_circles = circles_opt.copy()

    # Final polish: local optimization from best solution
    if best_circles is not None and time.time() - start_time < DEADLINE - 5:
        params_best = np.empty(3 * N)
        params_best[0::3] = best_circles[:, 0]
        params_best[1::3] = best_circles[:, 1]
        params_best[2::3] = best_circles[:, 2]

        lb = np.zeros(3 * N)
        ub = np.zeros(3 * N)
        ub[0::3] = 1.0
        ub[1::3] = 1.0
        ub[2::3] = 0.5

        try:
            # Polish with tighter tolerances
            result = minimize(
                _objective_and_grad,
                params_best,
                jac=True,
                method='L-BFGS-B',
                bounds=list(zip(lb, ub)),
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            x_p = result.x[0::3]
            y_p = result.x[1::3]
            r_p = result.x[2::3]
            circles_p = np.column_stack([x_p, y_p, r_p])
            circles_p_repaired = _repair(circles_p)

            if _is_valid(circles_p_repaired):
                sc = _score(circles_p_repaired)
                if sc > best_score:
                    best_score = sc
                    best_circles = circles_p_repaired.copy()

            if _is_valid(circles_p):
                sc = _score(circles_p)
                if sc > best_score:
                    best_score = sc
                    best_circles = circles_p.copy()
        except Exception:
            pass

    if best_circles is None:
        # Fallback to baseline
        return baseline

    return best_circles
