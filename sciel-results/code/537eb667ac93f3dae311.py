"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=100.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def smooth_objective_grad(params, alpha=100.0):
        """Smooth objective with analytical gradient for L-BFGS-B."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        n = len(x)

        # Compute distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        f = -np.sum(radii_soft)

        # Gradient computation
        grad = np.zeros(2 * n)

        for i in range(n):
            # Boundary gradient
            b_exp_sum = np.sum(exp_b[i, :])
            d_dxi = 0.0
            d_dyi = 0.0
            if b_exp_sum > 0:
                d_dxi = -(1/alpha) * (exp_b[i, 0] * (-alpha) + exp_b[i, 1] * alpha) / b_exp_sum
                d_dyi = -(1/alpha) * (exp_b[i, 2] * (-alpha) + exp_b[i, 3] * alpha) / b_exp_sum

            # Distance gradient
            d_exp_sum = np.sum(exp_scaled[i, :])
            if d_exp_sum > 0:
                d_dist_dxi = np.zeros(n)
                d_dist_dyi = np.zeros(n)
                for j in range(n):
                    if i == j:
                        continue
                    ddist = dist_matrix[i, j]
                    if ddist > 1e-10:
                        d_dist_dxi[j] = dx[i, j] / ddist
                        d_dist_dyi[j] = dy[i, j] / ddist

                d_dxi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dxi)) / d_exp_sum
                d_dyi += -(1/alpha) * np.sum(exp_scaled[i, :] * (-alpha * d_dist_dyi)) / d_exp_sum

            # Combine boundary and distance via soft-min
            ab_exp_sum = np.sum(exp_ab[i, :])
            if ab_exp_sum > 0:
                dr_dxi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dxi) + exp_ab[i, 1] * (-alpha * 0.5 * d_dxi)) / ab_exp_sum
                dr_dyi = -(1/alpha) * (exp_ab[i, 0] * (-alpha * d_dyi) + exp_ab[i, 1] * (-alpha * 0.5 * d_dyi)) / ab_exp_sum

                if radii_soft[i] > 0:
                    grad[2*i] = -dr_dxi
                    grad[2*i+1] = -dr_dyi

        return f, grad

    def push_apart(params, iterations=50):
        """Push overlapping circles apart to improve validity."""
        x = params[0::2].copy()
        y = params[1::2].copy()
        radii = compute_radii(x, y)

        for _ in range(iterations):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist and dist > 1e-10:
                        push = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        x[i] -= push * nx
                        y[i] -= push * ny
                        x[j] += push * nx
                        y[j] += push * ny
                        moved = True
            if not moved:
                break

        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        new_params = np.zeros(2 * N)
        new_params[0::2] = x
        new_params[1::2] = y
        return new_params

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        s = 1.0 / np.sqrt(2 * N)
        for row in range(8):
            for col in range(8):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                x0[idx] = np.clip(x0[idx], 0.05, 0.95)
                y0[idx] = np.clip(y0[idx], 0.05, 0.95)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def random_init():
        """Random initialization with some spacing."""
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def uniform_radius_init(r):
        """Initialize with all circles at radius r, placed to be non-overlapping."""
        # Try to place circles with minimum separation 2r
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        for i in range(N):
            for attempt in range(1000):
                x = np.random.uniform(r, 1-r)
                y = np.random.uniform(r, 1-r)
                ok = True
                for j in range(i):
                    dx = x - x0[j]
                    dy = y - y0[j]
                    if dx*dx + dy*dy < (2*r)**2 * 0.9:
                        ok = False
                        break
                if ok:
                    x0[i] = x
                    y0[i] = y
                    break
            else:
                x0[i] = np.random.uniform(r, 1-r)
                y0[i] = np.random.uniform(r, 1-r)
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    # Bisection on uniform radius to find best packing
    def bisection_uniform_radius(max_time=30):
        """Find best uniform radius packing via bisection."""
        best_params = None
        best_obj = np.inf
        t0 = time.time()

        # Binary search on radius
        lo = 0.0
        hi = 0.5  # maximum possible radius

        # First, find a feasible upper bound
        # Try decreasing radius until feasible
        for r in np.linspace(0.2, 0.05, 30):
            if time.time() - t0 > max_time:
                break
            params = uniform_radius_init(r)
            obj = objective(params)
            # Check if all radii are at least r
            x = params[0::2]
            y = params[1::2]
            radii = compute_radii(x, y)
            if np.min(radii) >= r * 0.95:
                # Feasible, try larger
                if obj < best_obj:
                    best_obj = obj
                    best_params = params.copy()
                hi = r
                break
        else:
            # No feasible radius found in range, use grid init
            best_params = grid_init()
            best_obj = objective(best_params)

        # Now refine
        if best_params is not None:
            for _ in range(15):
                if time.time() - t0 > max_time:
                    break
                mid = (lo + hi) / 2
                params = uniform_radius_init(mid)
                x = params[0::2]
                y = params[1::2]
                radii = compute_radii(x, y)
                if np.min(radii) >= mid * 0.95:
                    obj = objective(params)
                    if obj < best_obj:
                        best_obj = obj
                        best_params = params.copy()
                    lo = mid
                else:
                    hi = mid

        return best_params, best_obj

    # Main optimization
    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Try multiple initializations
    np.random.seed(42)
    inits = [grid_init(), hex_init()]
    for _ in range(4):
        inits.append(random_init())

    # Also try bisection uniform radius
    uni_params, uni_obj = bisection_uniform_radius(max_time=25)
    if uni_params is not None and uni_obj < best_obj:
        best_params = uni_params.copy()
        best_obj = uni_obj

    for init_params in inits:
        if time.time() > deadline - 60:
            break
        obj_val = objective(init_params)
        if obj_val < best_obj:
            best_params = init_params.copy()
            best_obj = obj_val

    # Multi-stage optimization with increasing alpha (smoothing sharpness)
    alphas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for i, alpha in enumerate(alphas):
        if time.time() > deadline - 30:
            break
            
        for restart in range(3):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                params_start = best_params.copy()
            elif restart == 1:
                params_start = best_params + np.random.randn(2*N) * 0.02
                params_start = np.clip(params_start, 0.01, 0.99)
            else:
                params_start = best_params + np.random.randn(2*N) * 0.05
                params_start = np.clip(params_start, 0.01, 0.99)
            
            try:
                ftol = 1e-10 if i < 2 else 1e-12
                res = minimize(
                    lambda p: smooth_objective_grad(p, alpha=alpha), 
                    params_start, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    jac=True,
                    options={'maxiter': 3000, 'ftol': ftol, 'gtol': 1e-7}
                )
                true_obj = objective(res.x)
                if true_obj < best_obj:
                    res_x = push_apart(res.x, iterations=30)
                    pa_obj = objective(res_x)
                    if pa_obj < best_obj:
                        best_params = res_x
                        best_obj = pa_obj
                    else:
                        best_params = res.x
                        best_obj = true_obj
            except Exception:
                pass

    # Powell refinement from best smooth result
    if time.time() < deadline - 25:
        try:
            cleaned = push_apart(best_params, iterations=50)
            clean_obj = objective(cleaned)
            if clean_obj < best_obj:
                best_params = cleaned
                best_obj = clean_obj

            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 8000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 10:
        try:
            cleaned = push_apart(best_params, iterations=100)
            clean_obj = objective(cleaned)
            if clean_obj < best_obj:
                best_params = cleaned
                best_obj = clean_obj

            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 15000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final cleanup and verification
    if time.time() < deadline - 3:
        try:
            final = push_apart(best_params, iterations=200)
            final_obj = objective(final)
            if final_obj < best_obj:
                best_params = final
                best_obj = final_obj
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.99999

    # Ensure circles are within bounds
    for i in range(N):
        if x[i] < radii[i]:
            x[i] = radii[i]
        if x[i] > 1 - radii[i]:
            x[i] = 1 - radii[i]
        if y[i] < radii[i]:
            y[i] = radii[i]
        if y[i] > 1 - radii[i]:
            y[i] = 1 - radii[i]

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
