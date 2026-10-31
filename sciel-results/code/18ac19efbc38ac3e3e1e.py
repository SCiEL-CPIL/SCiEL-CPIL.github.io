"""Circle packing in unit square - gradient-based optimization with smooth penalties."""
import numpy as np
import time

N = 26
_DEADLINE = 140.0  # safety margin before 150s kill


def _smooth_max(x, k=50.0):
    """Log-sum-exp approximation of max(x, 0)."""
    return (1.0 / k) * np.logaddexp(0.0, k * x)


def _pairwise_violation(pos, rad):
    """Compute overlap violations between all pairs (smooth)."""
    n = len(pos)
    dx = pos[:, 0:1] - pos[1:, 0:1].T
    dy = pos[:, 1:2] - pos[1:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    req = rad[:, None] + rad[None, :]
    return _smooth_max(req - dist)


def _boundary_violation(pos, rad):
    """Compute boundary violations (smooth)."""
    left = _smooth_max(rad - pos[:, 0])
    right = _smooth_max(rad - (1.0 - pos[:, 0]))
    bottom = _smooth_max(rad - pos[:, 1])
    top = _smooth_max(rad - (1.0 - pos[:, 1]))
    return left + right + bottom + top


def _objective_and_grad(theta, k=50.0):
    """
    Objective: maximize sum of radii subject to non-overlap and boundary constraints.
    theta = [x1,y1,r1, x2,y2,r2, ..., xn,yn,rn]
    Returns (value, grad) where value = -sum(r) + penalty.
    """
    n = N
    pos = theta[0:3*n:3].reshape(n, 2)
    rad = theta[1:3*n:3]

    # Penalty for violations
    pair_viol = _pairwise_violation(pos, rad, k)
    # Sum all pairwise violations
    pair_penalty = np.sum(pair_viol)

    bnd_penalty = np.sum(_boundary_violation(pos, rad, k))

    total_penalty = pair_penalty + bnd_penalty
    value = -np.sum(rad) + total_penalty

    # Gradient
    grad = np.zeros(3 * n)
    # d/dr: -1 + penalty gradient
    grad[1:3*n:3] = -1.0

    # Penalty gradient w.r.t. radii (boundary)
    bnd_grad = np.zeros(n)
    # left: d/dr = 1 if r > x (approx via sigmoid)
    left_arg = k * (rad - pos[:, 0])
    bnd_grad += 1.0 * (1.0 / (1.0 + np.exp(-left_arg)))
    right_arg = k * (rad - (1.0 - pos[:, 0]))
    bnd_grad += 1.0 * (1.0 / (1.0 + np.exp(-right_arg)))
    bottom_arg = k * (rad - pos[:, 1])
    bnd_grad += 1.0 * (1.0 / (1.0 + np.exp(-bottom_arg)))
    top_arg = k * (rad - (1.0 - pos[:, 1]))
    bnd_grad += 1.0 * (1.0 / (1.0 + np.exp(-top_arg)))
    grad[1:3*n:3] += bnd_grad

    # Penalty gradient w.r.t. radii (pairwise)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            req = rad[i] + rad[j]
            arg = k * (req - dist)
            sigmoid = 1.0 / (1.0 + np.exp(-arg))
            # d(softmax)/dr_i = sigmoid, d(softmax)/dr_j = sigmoid
            grad[3*i + 1] += sigmoid
            grad[3*j + 1] += sigmoid

    # Penalty gradient w.r.t. positions (boundary)
    # left: d/dx = -sigmoid, right: d/dx = +sigmoid
    left_s = 1.0 / (1.0 + np.exp(-k * (rad - pos[:, 0])))
    right_s = 1.0 / (1.0 + np.exp(-k * (rad - (1.0 - pos[:, 0]))))
    bottom_s = 1.0 / (1.0 + np.exp(-k * (rad - pos[:, 1])))
    top_s = 1.0 / (1.0 + np.exp(-k * (rad - (1.0 - pos[:, 1]))))
    grad[0:3*n:3] += -left_s + right_s
    grad[2:3*n:3] += -bottom_s + top_s

    # Penalty gradient w.r.t. positions (pairwise)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            req = rad[i] + rad[j]
            arg = k * (req - dist)
            sigmoid = 1.0 / (1.0 + np.exp(-arg))
            if dist > 1e-12:
                grad[3*i + 0] += sigmoid * dx / dist
                grad[3*i + 2] += sigmoid * dy / dist
                grad[3*j + 0] -= sigmoid * dx / dist
                grad[3*j + 2] -= sigmoid * dy / dist

    return value, grad


def _is_valid(result, tol=1e-6):
    """Check if result is valid: circles inside [0,1]^2 with disjoint interiors."""
    if result.shape != (N, 3):
        return False
    x, y, r = result[:, 0], result[:, 1], result[:, 2]
    if np.any(r <= 0) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _post_process(result):
    """Clamp positions and radii to valid range, shrink slightly if needed."""
    result = result.copy()
    for i in range(N):
        result[i, 2] = max(result[i, 2], 1e-8)
        result[i, 0] = np.clip(result[i, 0], result[i, 2], 1.0 - result[i, 2])
        result[i, 1] = np.clip(result[i, 1], result[i, 2], 1.0 - result[i, 2])
    # If still invalid, shrink radii
    if not _is_valid(result):
        for i in range(N):
            result[i, 2] *= 0.99
            result[i, 0] = np.clip(result[i, 0], result[i, 2], 1.0 - result[i, 2])
            result[i, 1] = np.clip(result[i, 1], result[i, 2], 1.0 - result[i, 2])
    return result


def _make_grid_initial():
    """Create a grid-based initial guess."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([cx, cy, r])
    return np.array(cells)


def _make_random_initial(seed=42):
    """Create a random initial guess with small circles."""
    rng = np.random.default_rng(seed)
    result = []
    for i in range(N):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.02, 0.05)
        result.append([x, y, r])
    return np.array(result)


def construct():
    start_time = time.time()
    best_result = None
    best_score = -np.inf

    # Start with grid initial
    init = _make_grid_initial()
    theta = init.flatten().copy()

    # Run L-BFGS-B with increasing k (smoothness parameter)
    from scipy.optimize import minimize

    k_schedule = [20.0, 40.0, 80.0, 150.0, 300.0]

    for k in k_schedule:
        if time.time() - start_time > _DEADLINE:
            break
        try:
            res = minimize(
                lambda t: _objective_and_grad(t, k)[0],
                theta,
                jac=lambda t: _objective_and_grad(t, k)[1],
                method='L-BFGS-B',
                bounds=[(1e-8, 1.0 - 1e-8)] * (3 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < _objective_and_grad(theta, k)[0]:
                theta = res.x
        except Exception:
            pass

    # Extract result
    pos = theta[0:3*N:3].reshape(N, 2)
    rad = theta[1:3*N:3]
    result = np.column_stack([pos, rad])
    result = _post_process(result)

    if _is_valid(result):
        best_result = result
        best_score = np.sum(result[:, 2])

    # Try a few random restarts if time permits
    for seed in [1, 7, 42, 123, 999]:
        if time.time() - start_time > _DEADLINE - 10:
            break
        init = _make_random_initial(seed)
        theta = init.flatten().copy()

        for k in k_schedule[:3]:  # fewer iterations for restarts
            if time.time() - start_time > _DEADLINE - 5:
                break
            try:
                res = minimize(
                    lambda t: _objective_and_grad(t, k)[0],
                    theta,
                    jac=lambda t: _objective_and_grad(t, k)[1],
                    method='L-BFGS-B',
                    bounds=[(1e-8, 1.0 - 1e-8)] * (3 * N),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.fun < _objective_and_grad(theta, k)[0]:
                    theta = res.x
            except Exception:
                pass

        pos = theta[0:3*N:3].reshape(N, 2)
        rad = theta[1:3*N:3]
        result = np.column_stack([pos, rad])
        result = _post_process(result)

        if _is_valid(result):
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    if best_result is None:
        # Fallback to grid
        best_result = _make_grid_initial()

    return best_result
