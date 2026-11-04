"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 135.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H, tol=1e-8):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=5000, tol=1e-10):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-6
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-8
                        dy = np.sin(angle) * 1e-8
                        dist = 1e-8
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-10
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-6, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _objective_and_grad(params, W, H, penalty_weight):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad.reshape(n, 3)[:, 2] -= 1.0  # d(-sum(r))/dr = -1

    # Smooth max function
    eps = 1e-4

    def smooth_max(v):
        return 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))

    def smooth_max_grad(v):
        return v / np.sqrt(v ** 2 + eps ** 2)

    # Boundary constraints:
    # x - r >= 0  => violation v1 = r - x
    # y - r >= 0  => violation v2 = r - y
    # x + r <= W  => violation v3 = x + r - W
    # y + r <= H  => violation v4 = y + r - H
    # r >= 0      => violation v5 = -r

    v1 = r - x
    v2 = r - y
    v3 = x + r - W
    v4 = y + r - H
    v5 = -r

    obj += penalty_weight * (np.sum(smooth_max(v1) ** 2) + np.sum(smooth_max(v2) ** 2) +
                             np.sum(smooth_max(v3) ** 2) + np.sum(smooth_max(v4) ** 2) +
                             np.sum(smooth_max(v5) ** 2))

    # Gradients for boundary penalties
    g1 = smooth_max_grad(v1)
    g2 = smooth_max_grad(v2)
    g3 = smooth_max_grad(v3)
    g4 = smooth_max_grad(v4)
    g5 = smooth_max_grad(v5)

    # d/dx: v1=-x, v3=+x
    # d/dr: v1=+r, v2=+r, v3=+r, v4=+r, v5=-r
    grad.reshape(n, 3)[:, 0] += penalty_weight * (-2 * smooth_max(v1) * g1 + 2 * smooth_max(v3) * g3)
    grad.reshape(n, 3)[:, 1] += penalty_weight * (-2 * smooth_max(v2) * g2 + 2 * smooth_max(v4) * g4)
    grad.reshape(n, 3)[:, 2] += penalty_weight * (2 * smooth_max(v1) * g1 + 2 * smooth_max(v2) * g2 +
                                                  2 * smooth_max(v3) * g3 + 2 * smooth_max(v4) * g4 -
                                                  2 * smooth_max(v5) * g5)

    # Overlap penalties: vectorized
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)

    r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum_mat - dist_mat

    iu, ju = np.triu_indices(n, k=1)
    v_vec = v_overlap[iu, ju]
    pos_v = np.maximum(v_vec, 0)
    obj += penalty_weight * np.sum(pos_v ** 2)

    # Gradient for overlap penalties
    # d(v)/dx_i = -dx_ij / dist_ij, d(v)/dx_j = +dx_ij / dist_ij
    # d(v)/dr_i = 1, d(v)/dr_j = 1
    grad_overlap_x = np.zeros(n)
    grad_overlap_y = np.zeros(n)
    grad_overlap_r = np.zeros(n)

    for idx in range(len(iu)):
        i = iu[idx]
        j = ju[idx]
        if pos_v[idx] > 0:
            dx = dx_mat[i, j]
            dy = dy_mat[i, j]
            dist = dist_mat[i, j]
            if dist > 1e-12:
                coeff = 2 * penalty_weight * pos_v[idx]
                grad_overlap_x[i] += -coeff * dx / dist
                grad_overlap_x[j] += coeff * dx / dist
                grad_overlap_y[i] += -coeff * dy / dist
                grad_overlap_y[j] += coeff * dy / dist
            grad_overlap_r[i] += 2 * penalty_weight * pos_v[idx]
            grad_overlap_r[j] += 2 * penalty_weight * pos_v[idx]

    grad.reshape(n, 3)[:, 0] += grad_overlap_x
    grad.reshape(n, 3)[:, 1] += grad_overlap_y
    grad.reshape(n, 3)[:, 2] += grad_overlap_r

    return obj, grad


def _local_optimize(W, H, initial_circles, time_budget=10.0):
    """
    Optimize using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = initial_circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = initial_circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    weights = [5, 20, 100, 500, 2000, 10000]
    current_params = params.copy()

    for w in weights:
        if time.time() - start > time_budget - 1:
            break
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, W, H, w)[0],
                current_params,
                jac=lambda p: _objective_and_grad(p, W, H, w)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
            )
            current_params = res.x
        except Exception:
            continue

        circles = current_params.reshape(N, 3)
        circles, valid = _separate_circles(circles, W, H, max_iter=1000)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score


def _high_precision_polish(circles, W, H, time_budget=5.0):
    """
    Final high-precision polish with very tight tolerances.
    """
    start = time.time()
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    # Use very high penalty weight for tight constraint enforcement
    for w in [50000, 200000]:
        if time.time() - start > time_budget - 0.5:
            break
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, W, H, w)[0],
                params,
                jac=lambda p: _objective_and_grad(p, W, H, w)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
            )
            params = res.x
        except Exception:
            continue

        circles_new = params.reshape(N, 3)
        circles_new, valid = _separate_circles(circles_new, W, H, max_iter=2000, tol=1e-12)
        if valid:
            score = np.sum(circles_new[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles_new.copy()
                params = best_circles.flatten()

    return best_circles, best_score


def _hexagonal_initialization(W, H):
    """
    Hexagonal packing initialization.
    """
    circles = []
    # Try to fit a hexagonal grid
    rows = 5
    cols = 5
    spacing_x = W / (cols + 0.5)
    spacing_y = H / (rows + 0.5)
    r = min(spacing_x, spacing_y) / 2 - 1e-4

    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= N:
                break
            x = (j + 0.5 + (0.5 if i % 2 == 0 else 0)) * spacing_x
            y = (i + 0.5) * spacing_y
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            circles.append([x, y, r])
            count += 1

    return np.array(circles[:N])


def _grid_initialization(W, H):
    """Grid initialization with 21 circles."""
    circles = []
    # 3x7 grid
    rows = 3
    cols = 7
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-4
            circles.append([cx, cy, r])
    return np.array(circles[:N])


def _random_initialization(W, H, rng):
    """Random initialization."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.03, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Aspect ratios to try
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Generate multiple initializations
        inits = []
        inits.append(_grid_initialization(W, H))
        inits.append(_hexagonal_initialization(W, H))
        inits.append(_random_initialization(W, H, rng))

        for init in inits:
            if time.time() - start_time > DEADLINE - 8:
                break
            remaining = DEADLINE - 8 - (time.time() - start_time)
            if remaining < 5:
                break
            time_budget = min(12.0, remaining / 2.0)

            circles, score = _local_optimize(W, H, init, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = W, H

    # Final high-precision polish
    if best_result is not None and time.time() - start_time < DEADLINE - 6:
        remaining = DEADLINE - 6 - (time.time() - start_time)
        if remaining > 3:
            polished, score = _high_precision_polish(best_result, best_W, best_H, time_budget=min(5.0, remaining))
            if score > best_score:
                best_score = score
                best_result = polished.copy()

    # Fallback
    if best_result is None:
        best_result = _grid_initialization(1.0, 1.0)
        best_result, _ = _separate_circles(best_result, 1.0, 1.0)
        best_W, best_H = 1.0, 1.0

    # Final validity check and repair
    if not _check_validity(best_result, best_W, best_H):
        repaired, valid = _separate_circles(best_result, best_W, best_H, max_iter=10000)
        if valid:
            best_result = repaired

    return best_result
# EVOLVE-BLOCK-END
