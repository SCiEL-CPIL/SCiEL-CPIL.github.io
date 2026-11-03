import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time
import random

N = 21
_EPS = 1e-9


def _softplus(z, beta=100.0):
    """Smooth max(0, z)."""
    return np.where(z > 0, z + np.log1p(np.exp(-beta * z)) / beta,
                    np.log1p(np.exp(beta * z)) / beta)


def _softplus_grad(z, beta=100.0):
    """Gradient of softplus w.r.t. z."""
    return np.where(z > 0, 1.0, 1.0 / (1.0 + np.exp(-beta * z)))


def _delaunay_pairs(x, y):
    """Return list of (i, j) pairs from Delaunay triangulation of centers."""
    pts = np.column_stack((x, y))
    try:
        tri = Delaunay(pts)
        pairs = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    pairs.add((min(i, j), max(i, j)))
        return list(pairs)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _objective_and_grad(state, beta=100.0, var_weight=0.0):
    """
    Smooth objective: maximize sum(r) - penalties.
    Returns (loss, grad) where loss = -score, so minimizing loss maximizes score.
    """
    W = state[0]
    W_c = np.clip(W, 0.2, 1.8)
    H = 2.0 - W_c
    coords = state[1:].reshape(N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    r = coords[:, 2]

    # --- Boundary softplus penalties ---
    # left: r - x should be <= 0
    viol_l = r - x
    viol_r = x + r - W_c
    viol_b = r - y
    viol_t = y + r - H
    bpen = _softplus(viol_l, beta) + _softplus(viol_r, beta) + _softplus(viol_b, beta) + _softplus(viol_t, beta)

    # --- Overlap softplus penalties using Delaunay pairs ---
    pairs = _delaunay_pairs(x, y)
    open = 0.0
    open_grad_x = np.zeros(N)
    open_grad_y = np.zeros(N)
    open_grad_r = np.zeros(N)
    for (i, j) in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy) + 1e-15
        gap = dist - (r[i] + r[j])  # should be >= 0
        z = -gap  # violation when gap < 0 => z > 0
        sp = _softplus(z, beta)
        open += sp
        g = _softplus_grad(z, beta)
        if g > 1e-12:
            # d(sp)/d(x_i) = g * d(z)/d(x_i) = g * (-dx/dist)
            # d(z)/d(x_i) = -dx/dist, d(z)/d(x_j) = dx/dist
            open_grad_x[i] += g * (-dx / dist)
            open_grad_x[j] += g * (dx / dist)
            open_grad_y[i] += g * (-dy / dist)
            open_grad_y[j] += g * (dy / dist)
            # d(z)/d(r_i) = -1, d(z)/d(r_j) = -1
            open_grad_r[i] += g * (-1.0)
            open_grad_r[j] += g * (-1.0)

    # --- Small radius penalty ---
    rpen = _softplus(0.01 - r, beta)

    # --- Variance penalty to encourage equal radii ---
    vpen = 0.0
    vpen_grad_r = np.zeros(N)
    if var_weight > 0:
        mean_r = np.mean(r)
        vpen = var_weight * np.sum((r - mean_r) ** 2)
        vpen_grad_r = 2.0 * var_weight * (r - mean_r)

    score = np.sum(r) - 50.0 * np.sum(bpen) - 100.0 * open - 10.0 * np.sum(rpen) - vpen
    loss = -score

    # --- Gradients ---
    grad = np.zeros(1 + 3 * N)
    # d(score)/d(W) via boundary terms:
    # viol_r = x + r - W_c, d(viol_r)/dW = -1 (when W in [0.2,1.8])
    # viol_t = y + r - H, H = 2-W, d(viol_t)/dW = +1
    dW = 0.0
    if 0.2 < W < 1.8:
        for i in range(N):
            z_r = x[i] + r[i] - W_c
            g_r = _softplus_grad(z_r, beta)
            dW += -50.0 * g_r  # d(bpen)/dW contribution from right
            z_t = y[i] + r[i] - H
            g_t = _softplus_grad(z_t, beta)
            dW += -50.0 * g_t  # d(bpen)/dW contribution from top (dH/dW = -1, so d(viol_t)/dW = +1)
    grad[0] = -dW  # d(loss)/dW = -d(score)/dW

    for i in range(N):
        base = 1 + 3 * i
        # x gradient
        gx = 0.0
        # boundary left: d(viol_l)/dx = -1
        gx += -50.0 * _softplus_grad(r[i] - x[i], beta) * (-1.0)
        # boundary right: d(viol_r)/dx = +1
        gx += -50.0 * _softplus_grad(x[i] + r[i] - W_c, beta) * (1.0)
        # overlap
        gx += -100.0 * open_grad_x[i]
        grad[base] = -gx  # d(loss)/dx = -d(score)/dx

        # y gradient
        gy = 0.0
        gy += -50.0 * _softplus_grad(r[i] - y[i], beta) * (-1.0)
        gy += -50.0 * _softplus_grad(y[i] + r[i] - H, beta) * (1.0)
        gy += -100.0 * open_grad_y[i]
        grad[base + 1] = -gy

        # r gradient
        gr = 1.0  # d(sum_r)/dr = 1
        gr += -50.0 * _softplus_grad(r[i] - x[i], beta) * (1.0)
        gr += -50.0 * _softplus_grad(x[i] + r[i] - W_c, beta) * (1.0)
        gr += -50.0 * _softplus_grad(r[i] - y[i], beta) * (1.0)
        gr += -50.0 * _softplus_grad(y[i] + r[i] - H, beta) * (1.0)
        gr += -100.0 * open_grad_r[i]
        gr += -10.0 * _softplus_grad(0.01 - r[i], beta) * (-1.0)
        gr += -vpen_grad_r[i]
        grad[base + 2] = -gr

    return loss, grad


def _strictly_valid(state, eps=1e-7):
    W = state[0]
    W = np.clip(W, 0.2, 1.8)
    H = 2.0 - W
    coords = state[1:].reshape(N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    r = coords[:, 2]
    if np.any(r < 1e-6):
        return False
    if np.any(x - r < -eps) or np.any(x + r > W + eps):
        return False
    if np.any(y - r < -eps) or np.any(y + r > H + eps):
        return False
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    min_dist = r[idx_i] + r[idx_j] - eps
    if np.any(dist_sq < min_dist * min_dist):
        return False
    return True


def _fast_valid(state, eps=1e-7):
    """Fast validity check using Delaunay pairs only."""
    W = state[0]
    W = np.clip(W, 0.2, 1.8)
    H = 2.0 - W
    coords = state[1:].reshape(N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    r = coords[:, 2]
    if np.any(r < 1e-6):
        return False
    if np.any(x - r < -eps) or np.any(x + r > W + eps):
        return False
    if np.any(y - r < -eps) or np.any(y + r > H + eps):
        return False
    pairs = _delaunay_pairs(x, y)
    for (i, j) in pairs:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx * dx + dy * dy
        min_dist = r[i] + r[j] - eps
        if dist_sq < min_dist * min_dist:
            return False
    return True


def _hex_init(W_init=1.0, jitter=0.0):
    W = np.clip(W_init, 0.2, 1.8)
    H = 2.0 - W
    state = np.zeros(1 + 3 * N)
    state[0] = W

    r_est = np.sqrt(0.9069 * W * H / (N * np.pi))
    r = r_est
    positions = []

    for attempt in range(30):
        positions = []
        y = r
        row_idx = 0
        while y <= H - r + 1e-9:
            x_start = r if row_idx % 2 == 0 else r + r
            x = x_start
            while x <= W - r + 1e-9:
                positions.append((x, y, r))
                x += 2.0 * r
            y += np.sqrt(3) * r
            row_idx += 1
        if len(positions) >= N:
            break
        r *= 0.95

    if len(positions) < N:
        positions = []
        side = 5
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                gx = (i + 0.5) / side * W
                gy = (j + 0.5) / side * H
                r_g = 0.5 / side * min(W, H) * 0.9
                positions.append((gx, gy, r_g))

    for i in range(N):
        if i < len(positions):
            px, py, pr = positions[i]
            if jitter > 0:
                px += random.uniform(-jitter, jitter)
                py += random.uniform(-jitter, jitter)
            state[1 + 3 * i] = px
            state[1 + 3 * i + 1] = py
            state[1 + 3 * i + 2] = pr
        else:
            state[1 + 3 * i] = W / 2
            state[1 + 3 * i + 1] = H / 2
            state[1 + 3 * i + 2] = 0.01
    return state


def _grid_init(W_init=1.0, jitter=0.0):
    W = np.clip(W_init, 0.2, 1.8)
    H = 2.0 - W
    state = np.zeros(1 + 3 * N)
    state[0] = W

    best_positions = None
    for rows in range(3, 8):
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            continue
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                gx = (j + 0.5) / cols * W
                gy = (i + 0.5) / rows * H
                r_g = 0.5 * min(W / cols, H / rows) * 0.85
                positions.append((gx, gy, r_g))
        if len(positions) >= N:
            mean_r = np.mean([p[2] for p in positions])
            if best_positions is None or mean_r > np.mean([p[2] for p in best_positions]):
                best_positions = positions

    if best_positions is None:
        side = 5
        best_positions = []
        for i in range(side):
            for j in range(side):
                if len(best_positions) >= N:
                    break
                gx = (i + 0.5) / side * W
                gy = (j + 0.5) / side * H
                r_g = 0.5 / side * min(W, H) * 0.9
                best_positions.append((gx, gy, r_g))

    for i in range(N):
        px, py, pr = best_positions[i]
        if jitter > 0:
            px += random.uniform(-jitter, jitter)
            py += random.uniform(-jitter, jitter)
        state[1 + 3 * i] = px
        state[1 + 3 * i + 1] = py
        state[1 + 3 * i + 2] = pr
    return state


def _random_init(W_init=1.0):
    W = np.clip(W_init, 0.2, 1.8)
    H = 2.0 - W
    state = np.zeros(1 + 3 * N)
    state[0] = W

    positions = []
    r_est = np.sqrt(0.9069 * W * H / (N * np.pi))

    for i in range(N):
        placed = False
        for attempt in range(2000):
            r = r_est * random.uniform(0.7, 1.3)
            px = random.uniform(r, W - r)
            py = random.uniform(r, H - r)
            ok = True
            for j, (qx, qy, qr) in enumerate(positions):
                dx = px - qx
                dy = py - qy
                if dx * dx + dy * dy < (r + qr) ** 2:
                    ok = False
                    break
            if ok:
                positions.append((px, py, r))
                placed = True
                break
        if not placed:
            positions.append((W / 2 + random.uniform(-0.1, 0.1),
                              H / 2 + random.uniform(-0.1, 0.1),
                              r_est * 0.5))

    for i in range(N):
        px, py, pr = positions[i]
        state[1 + 3 * i] = px
        state[1 + 3 * i + 1] = py
        state[1 + 3 * i + 2] = pr
    return state


def _polish_gradient(state, max_iter=500, beta=100.0, var_weight=0.0, time_limit=None):
    """
    Custom gradient descent with line search for the smooth objective.
    """
    x = state.copy()
    best_loss, _ = _objective_and_grad(x, beta, var_weight)
    best_x = x.copy()
    step_size = 0.01
    start_t = time.time()

    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        loss, grad = _objective_and_grad(x, beta, var_weight)
        if loss < best_loss - 1e-10:
            best_loss = loss
            best_x = x.copy()

        # Gradient norm check
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break

        # Line search (backtracking)
        alpha = step_size
        for ls in range(20):
            x_try = x - alpha * grad
            # Clamp W
            x_try[0] = np.clip(x_try[0], 0.2, 1.8)
            # Clamp coords
            W_t = np.clip(x_try[0], 0.2, 1.8)
            H_t = 2.0 - W_t
            coords = x_try[1:].reshape(N, 3)
            coords[:, 0] = np.clip(coords[:, 0], 0, W_t)
            coords[:, 1] = np.clip(coords[:, 1], 0, H_t)
            coords[:, 2] = np.clip(coords[:, 2], 1e-6, min(W_t, H_t) / 2)
            x_try[1:] = coords.flatten()

            loss_try, _ = _objective_and_grad(x_try, beta, var_weight)
            if loss_try < loss - 1e-12:
                x = x_try
                break
            alpha *= 0.5
        else:
            # No improvement, shrink step
            step_size *= 0.5
            if step_size < 1e-8:
                break

    if best_loss < loss:
        return best_x, best_loss
    return x, loss


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Generate multiple initial configurations
    configs = []
    W_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

    for W_init in W_values:
        configs.append(_hex_init(W_init, jitter=0.0))
        configs.append(_grid_init(W_init, jitter=0.0))

    configs.append(_random_init(1.0))
    configs.append(_random_init(0.9))
    configs.append(_random_init(1.1))

    # Jittered versions
    for W_init in [0.9, 1.0, 1.1]:
        configs.append(_hex_init(W_init, jitter=0.03))
        configs.append(_grid_init(W_init, jitter=0.03))

    best_score = -np.inf
    best_state = None

    # Phase 1: Gradient polish from each config
    for idx, p0 in enumerate(configs):
        if time.time() > deadline - 15:
            break

        try:
            # Use moderate beta and small variance weight
            polished, loss = _polish_gradient(p0, max_iter=300, beta=80.0,
                                              var_weight=0.1,
                                              time_limit=deadline - 15)
            score = -loss
            # Validate
            if _fast_valid(polished):
                if score > best_score:
                    best_score = score
                    best_state = polished.copy()
        except Exception:
            pass

    # Phase 2: Fine-tune with higher beta and stronger variance penalty
    if best_state is not None and time.time() < deadline - 20:
        try:
            polished, loss = _polish_gradient(best_state, max_iter=400, beta=150.0,
                                              var_weight=0.5,
                                              time_limit=deadline - 20)
            score = -loss
            if _fast_valid(polished) and score > best_score:
                best_score = score
                best_state = polished.copy()
        except Exception:
            pass

    # Phase 3: Final polish with very high beta
    if best_state is not None and time.time() < deadline - 10:
        try:
            polished, loss = _polish_gradient(best_state, max_iter=200, beta=300.0,
                                              var_weight=0.0,
                                              time_limit=deadline - 10)
            score = -loss
            if _fast_valid(polished) and score > best_score:
                best_score = score
                best_state = polished.copy()
        except Exception:
            pass

    # Fallback
    if best_state is None:
        W = 1.0
        H = 1.0
        side = 5
        result = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                gx = (i + 0.5) / side * W
                gy = (j + 0.5) / side * H
                r = 0.5 / side * min(W, H) * 0.9
                result.append([gx, gy, r])
                count += 1
        return np.array(result)

    # Final strict validation
    if not _strictly_valid(best_state):
        # Reduce radii progressively
        coords = best_state[1:].reshape(N, 3)
        for factor in [0.95, 0.9, 0.85, 0.8]:
            coords[:, 2] *= factor
            best_state[1:] = coords.flatten()
            if _strictly_valid(best_state):
                break
        if not _strictly_valid(best_state):
            # Last resort fallback
            W = 1.0
            H = 1.0
            side = 5
            result = []
            count = 0
            for i in range(side):
                for j in range(side):
                    if count >= N:
                        break
                    gx = (i + 0.5) / side * W
                    gy = (j + 0.5) / side * H
                    r = 0.5 / side * min(W, H) * 0.9
                    result.append([gx, gy, r])
                    count += 1
            return np.array(result)

    # Extract final result
    W = np.clip(best_state[0], 0.2, 1.8)
    H = 2.0 - W
    coords = best_state[1:].reshape(N, 3)

    # Clamp to valid range
    x = np.clip(coords[:, 0], 0, W)
    y = np.clip(coords[:, 1], 0, H)
    r = np.clip(coords[:, 2], 1e-6, min(W, H) / 2)

    result = np.column_stack((x, y, r))

    # Final validity check
    final_state = np.zeros(1 + 3 * N)
    final_state[0] = W
    final_state[1:] = result.flatten()
    if not _strictly_valid(final_state):
        result[:, 2] *= 0.9
        final_state[1:] = result.flatten()
        if not _strictly_valid(final_state):
            result[:, 2] *= 0.9
            final_state[1:] = result.flatten()
            if not _strictly_valid(final_state):
                W = 1.0
                H = 1.0
                side = 5
                result = []
                count = 0
                for i in range(side):
                    for j in range(side):
                        if count >= N:
                            break
                        gx = (i + 0.5) / side * W
                        gy = (j + 0.5) / side * H
                        r = 0.5 / side * min(W, H) * 0.9
                        result.append([gx, gy, r])
                        count += 1
                result = np.array(result)

    return result
