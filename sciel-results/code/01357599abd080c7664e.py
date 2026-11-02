"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21


def _softplus(x, eps=1e-8):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -30, 20))))


def _objective_and_grad(params):
    """
    params: [W, H, x_0..x_20, y_0..y_20, r_0..r_20]
    Total length = 2 + 2*21 + 21 = 65
    """
    W = params[0]
    H = params[1]
    x = params[2:2+N]
    y = params[2+N:2+2*N]
    r = params[2+2*N:]

    # Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2
    # We'll use a penalty for W+H > 2
    perim_penalty = _softplus((W + H - 2.0) * 100.0)

    # Bounds penalty: circles must be inside rectangle
    bounds_penalty = np.zeros(N)
    for i in range(N):
        bounds_penalty[i] = (
            _softplus((r[i] - x[i]) * 100.0) +
            _softplus((r[i] - (W - x[i])) * 100.0) +
            _softplus((r[i] - y[i]) * 100.0) +
            _softplus((r[i] - (H - y[i])) * 100.0)
        )

    # Overlap penalty: for each pair (i,j), d_ij < r_i + r_j
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            violation = r[i] + r[j] - d
            overlap_penalty += _softplus(violation * 100.0)

    # Positive radius penalty
    radius_penalty = np.sum(_softplus(-r * 100.0))

    total_penalty = perim_penalty + np.sum(bounds_penalty) + overlap_penalty + radius_penalty
    value = np.sum(r) - total_penalty

    return -value  # minimize negative value


def _objective_and_grad_fn(params):
    """Returns (objective, gradient) for L-BFGS-B."""
    W = params[0]
    H = params[1]
    x = params[2:2+N].copy()
    y = params[2+N:2+2*N].copy()
    r = params[2+2*N:].copy()

    perim_violation = W + H - 2.0
    perim_grad_W = 1.0
    perim_grad_H = 1.0
    # softplus derivative: sigmoid(x*100)*100
    perim_factor = 100.0 / (1.0 + np.exp(-np.clip(perim_violation * 100.0, -30, 20)))
    perim_pen = perim_factor * perim_violation if perim_violation > 0 else np.log1p(np.exp(perim_violation*100.0)) if perim_violation > -20 else 0.0
    # Actually let's just compute the full thing numerically for safety, or do it carefully

    # Let me use a cleaner analytical approach
    def sp(val):
        if val > 20:
            return val
        elif val < -20:
            return 0.0
        return np.log1p(np.exp(val))

    def sp_grad(val):
        if val > 20:
            return 1.0
        elif val < -20:
            return 0.0
        return 1.0 / (1.0 + np.exp(-val))

    # Objective: maximize sum(r) - penalties
    # Penalties:
    # 1. Perimeter: softplus((W+H-2)*100)
    # 2. Bounds: for each circle, softplus((r-x)*100) + softplus((r-(W-x))*100) + softplus((r-y)*100) + softplus((r-(H-y))*100)
    # 3. Overlap: softplus((r_i+r_j-d)*100) for each pair
    # 4. Negative radius: softplus(-r*100)

    # We want to MINIMIZE: -sum(r) + all_penalties
    # So d(objective)/d(params) = -d(sum(r))/d(params) + d(penalties)/d(params)

    obj = -np.sum(r)

    grad = np.zeros(len(params))

    # Perimeter penalty
    pv = (W + H - 2.0) * 100.0
    obj += sp(pv)
    pg = sp_grad(pv) * 100.0
    grad[0] += pg
    grad[1] += pg

    # Bounds penalties
    for i in range(N):
        ri = r[i]
        xi = x[i]
        yi = y[i]

        # Left: softplus((r - x) * 100)
        v1 = (ri - xi) * 100.0
        obj += sp(v1)
        g1 = sp_grad(v1) * 100.0
        grad[2+i] += -g1       # d/dx_i
        grad[2+2*N+i] += g1    # d/dr_i

        # Right: softplus((r - (W-x)) * 100) = softplus((r-W+x)*100)
        v2 = (ri - W + xi) * 100.0
        obj += sp(v2)
        g2 = sp_grad(v2) * 100.0
        grad[0] += -g2         # d/dW
        grad[2+i] += g2        # d/dx_i
        grad[2+2*N+i] += g2    # d/dr_i

        # Bottom: softplus((r - y) * 100)
        v3 = (ri - yi) * 100.0
        obj += sp(v3)
        g3 = sp_grad(v3) * 100.0
        grad[2+N+i] += -g3     # d/dy_i
        grad[2+2*N+i] += g3    # d/dr_i

        # Top: softplus((r - (H-y)) * 100) = softplus((r-H+y)*100)
        v4 = (ri - H + yi) * 100.0
        obj += sp(v4)
        g4 = sp_grad(v4) * 100.0
        grad[1] += -g4         # d/dH
        grad[2+N+i] += g4      # d/dy_i
        grad[2+2*N+i] += g4    # d/dr_i

    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-10)
            v = (r[i] + r[j] - d) * 100.0
            obj += sp(v)
            g = sp_grad(v) * 100.0
            # d/dr_i, d/dr_j
            grad[2+2*N+i] += g
            grad[2+2*N+j] += g
            # d/dx_i, d/dx_j
            if d > 1e-10:
                gdx = -g * dx / d
                gdy = -g * dy / d
            else:
                gdx = 0.0
                gdy = 0.0
            grad[2+i] += gdx
            grad[2+j] -= gdx
            grad[2+N+i] += gdy
            grad[2+N+j] -= gdy

    # Negative radius penalty
    for i in range(N):
        v = -r[i] * 100.0
        obj += sp(v)
        g = sp_grad(v) * 100.0
        grad[2+2*N+i] += -g  # d/dr_i of softplus(-r*100) = -100*sigmoid(-r*100)

    # The -sum(r) part
    for i in range(N):
        grad[2+2*N+i] += -1.0

    return obj, grad


def _validate(W, H, x, y, r, tol=1e-5):
    """Check if configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if 2*(W+H) > 4 + tol:
        return False
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            if d2 < (r[i]+r[j])**2 - tol:
                return False
    return True


def _make_initial(x0, y0, r0, W0, H0, scale=1.0):
    """Create initial parameter vector with jitter."""
    params = np.array([W0, H0])
    x = x0 + scale * np.random.randn(N)
    y = y0 + scale * np.random.randn(N)
    r = r0 + scale * 0.1 * np.random.randn(N)
    r = np.maximum(r, 0.01)
    params = np.concatenate([params, x, y, r])
    return params


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_val = -np.inf
    best_params = None

    # Seed configuration: 5x5 grid in a square of side 1 (perimeter 4)
    side = 5
    W0 = 1.0
    H0 = 1.0
    x0 = np.array([(i + 0.5) / side for i in range(side)] * 5)[:N]
    y0 = np.array([(j + 0.5) / side for j in range(side) for i in range(side)][:N])
    r0 = np.full(N, 0.5 / side - 0.01)

    # Try different initial configurations
    init_configs = []

    # 1. Grid with various aspect ratios
    for W0_try in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        H0_try = 2.0 - W0_try
        if H0_try <= 0:
            continue
        x0_try = np.array([(i + 0.5) / 5 for i in range(5)] * 5)[:N] * (W0_try / 1.0)
        y0_try = np.array([(j + 0.5) / 5 for j in range(5) for i in range(5)][:N]) * (H0_try / 1.0)
        r0_try = np.full(N, 0.1)
        init_configs.append((W0_try, H0_try, x0_try, y0_try, r0_try))

    # 2. Hexagonal-like packing
    for W0_try in [1.0, 1.2, 1.4]:
        H0_try = 2.0 - W0_try
        if H0_try <= 0:
            continue
        xs = []
        ys = []
        for row in range(5):
            for col in range(5):
                if len(xs) >= N:
                    break
                xs.append((col + 0.5) / 5 * W0_try + (row % 2) * W0_try * 0.05)
                ys.append((row + 0.5) / 5 * H0_try)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        r0_try = np.full(N, min(W0_try, H0_try) * 0.08)
        init_configs.append((W0_try, H0_try, xs, ys, r0_try))

    # 3. Random initial configs
    for _ in range(10):
        W0_try = np.random.uniform(0.5, 1.5)
        H0_try = 2.0 - W0_try
        if H0_try <= 0.2:
            continue
        x0_try = np.random.uniform(0.1, W0_try - 0.1, N)
        y0_try = np.random.uniform(0.1, H0_try - 0.1, N)
        r0_try = np.full(N, np.random.uniform(0.03, 0.08))
        init_configs.append((W0_try, H0_try, x0_try, y0_try, r0_try))

    for cfg_idx, (W0, H0, x0, y0, r0) in enumerate(init_configs):
        if time.time() > deadline - 5.0:
            break

        # Multiple restarts per config
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break

            scale = 0.02 * (restart + 1)
            params = _make_initial(x0, y0, r0, W0, H0, scale)

            # Bounds for L-BFGS-B
            W_min, W_max = 0.1, 2.0
            H_min, H_max = 0.1, 2.0
            r_min = 0.001
            r_max = 1.0
            x_min = np.full(N, 0.0)
            x_max = np.full(N, 2.0)
            y_min = np.full(N, 0.0)
            y_max = np.full(N, 2.0)

            bounds = [
                (W_min, W_max), (H_min, H_max)
            ]
            for i in range(N):
                bounds.append((x_min[i], x_max[i]))
            for i in range(N):
                bounds.append((y_min[i], y_max[i]))
            for i in range(N):
                bounds.append((r_min, r_max))

            try:
                res = minimize(
                    _objective_and_grad_fn,
                    params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'maxfun': 2000}
                )
            except Exception:
                continue

            W = res.x[0]
            H = res.x[1]
            x = res.x[2:2+N]
            y = res.x[2+N:2+2*N]
            r = res.x[2+2*N:]

            if _validate(W, H, x, y, r):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()

        # Polish: run more iterations on best found so far
        if best_params is not None and time.time() < deadline - 10.0:
            try:
                res2 = minimize(
                    _objective_and_grad_fn,
                    best_params,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10, 'maxfun': 5000}
                )
                W = res2.x[0]
                H = res2.x[1]
                x = res2.x[2:2+N]
                y = res2.x[2+N:2+2*N]
                r = res2.x[2+2*N:]
                if _validate(W, H, x, y, r):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_params = res2.x.copy()
            except Exception:
                pass

    # Fallback to seed if nothing found
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    W = best_params[0]
    H = best_params[1]
    x = best_params[2:2+N]
    y = best_params[2+N:2+2*N]
    r = best_params[2+2*N:]

    # Final validation and return
    if not _validate(W, H, x, y, r, tol=1e-4):
        # Try to fix by clamping
        for i in range(N):
            r[i] = max(r[i], 0.001)
            x[i] = np.clip(x[i], r[i], W - r[i]) if W > 2*r[i] else W/2
            y[i] = np.clip(y[i], r[i], H - r[i]) if H > 2*r[i] else H/2

    result = np.column_stack((x, y, r))
    return result
