"""Circle packing in unit square, n=32: maximize sum of radii via smooth-penalty L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32
_DEADLINE = 150.0
_MARGIN = 10.0          # safety margin before the hard kill
_SEED = 12345


def _initial_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    r0 = 0.5 / side - 1e-3
    return np.array(cells[:n]), np.full(n, r0)


def _smooth_penalty_grad(pos, r, beta):
    """Return (penalty, grad_pos, grad_r) for overlap + boundary constraints."""
    n = len(r)
    x, y = pos[:, 0], pos[:, 1]
    pen = 0.0
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)

    # Overlap penalty: exp(beta * max(0, r_i + r_j - d_ij))
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            gap = r[i] + r[j] - d
            if gap > 0.0:
                e = np.exp(beta * gap)
                pen += e
                if d > 1e-9:
                    # d/|d| = (x_i - x_j)/d
                    ux = dx / d
                    uy = dy / d
                    # d gap / d x_i = -ux ; d gap / d r_i = 1
                    coef = beta * e
                    gx[i] += -coef * ux
                    gy[i] += -coef * uy
                    gx[j] += coef * ux
                    gy[j] += coef * uy
                    gr[i] += coef
                    gr[j] += coef
                else:
                    # coincident centers: nudge apart deterministically
                    coef = beta * e
                    gx[i] += coef
                    gr[i] += coef
                    gx[j] -= coef
                    gr[j] += coef

    # Boundary penalty: exp(beta * max(0, r_i - margin_i))
    for i in range(n):
        margins = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        m = margins.min()
        over = r[i] - m
        if over > 0.0:
            e = np.exp(beta * over)
            pen += e
            coef = beta * e
            gr[i] += coef
            # gradient w.r.t. position: push toward the tightest wall
            idx = int(np.argmin(margins))
            if idx == 0:      # x < 0 side
                gx[i] += coef
            elif idx == 1:    # x > 1 side
                gx[i] -= coef
            elif idx == 2:    # y < 0 side
                gy[i] += coef
            else:             # y > 1 side
                gy[i] -= coef

    return pen, gx, gy, gr


def _objective(params, beta):
    n = len(params) // 3
    pos = params[:2 * n].reshape(n, 2)
    r = np.exp(params[2 * n:])
    pen, gx, gy, gr = _smooth_penalty_grad(pos, r, beta)
    total = r.sum() + pen
    # gradient
    grad = np.zeros_like(params)
    grad[:n] = gx
    grad[n:2 * n] = gy
    grad[2 * n:] = r * gr - r  # d r.sum()/d log_r = r ; d pen/d log_r = r * gr
    return total, grad


def _is_valid(pos, r, tol=1e-6):
    n = len(r)
    if np.any(r <= 0):
        return False
    if np.any(pos < -tol) or np.any(pos > 1.0 + tol):
        return False
    # boundary: circle must be inside
    if np.any(r > pos[:, 0] + tol) or np.any(r > (1.0 - pos[:, 0]) + tol):
        return False
    if np.any(r > pos[:, 1] + tol) or np.any(r > (1.0 - pos[:, 1]) + tol):
        return False
    # overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _make_initial(rng, mode):
    """Return params (x, y, log_r) for a given random mode."""
    if mode == 0:
        # grid-like with jitter
        pos, r0 = _initial_grid(N)
        pos = pos + rng.uniform(-0.05, 0.05, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        r = r0 * np.exp(rng.uniform(-0.1, 0.1, N))
    elif mode == 1:
        # random spread
        pos = rng.uniform(0.1, 0.9, (N, 2))
        r = np.full(N, 0.5 / np.ceil(np.sqrt(N)) * 0.8)
    elif mode == 2:
        # clustered: a few big + many small
        pos = rng.uniform(0.15, 0.85, (N, 2))
        r = np.full(N, 0.05)
        for k in range(4):
            r[k] = rng.uniform(0.12, 0.18)
    else:
        pos = rng.uniform(0.1, 0.9, (N, 2))
        r = rng.uniform(0.04, 0.1, N)
    log_r = np.log(np.clip(r, 1e-4, None))
    params = np.concatenate([pos.ravel(), log_r])
    return params


def _optimize_once(params0, beta_schedule, max_iter, deadline):
    """Run L-BFGS-B with a ramped beta schedule. Returns best (pos, r, score)."""
    best_params = params0.copy()
    best_val = None
    # beta ramp: low to high
    n_stages = len(beta_schedule)
    iters_per = max(1, max_iter // n_stages)
    for s, beta in enumerate(beta_schedule):
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, best_params, args=(beta,),
                           method='L-BFGS-B', jac=True,
                           options={'maxiter': iters_per, 'gtol': 1e-6})
            if res.success or (best_val is None) or res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except Exception:
            break
    n = len(best_params) // 3
    pos = best_params[:2 * n].reshape(n, 2)
    r = np.exp(best_params[2 * n:])
    return pos, r, best_val


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    rng = np.random.default_rng(_SEED)

    # baseline grid solution
    pos0, r0 = _initial_grid(N)
    best_score = r0.sum()
    best_out = np.column_stack([pos0, r0])

    beta_schedule = [20.0, 50.0, 120.0, 300.0]
    max_iter = 300
    n_restarts = 12

    for it in range(n_restarts):
        if time.time() > deadline:
            break
        mode = it % 4
        params0 = _make_initial(rng, mode)
        pos, r, val = _optimize_once(params0, beta_schedule, max_iter, deadline)
        if not _is_valid(pos, r):
            continue
        score = r.sum()
        if score > best_score:
            best_score = score
            best_out = np.column_stack([pos, r])

    # final high-precision polish on the best solution
    if time.time() < deadline - 2.0 and best_out is not None:
        p = best_out[:, 0].copy()
        q = best_out[:, 1].copy()
        rr = best_out[:, 2].copy()
        params0 = np.concatenate([p, q, np.log(np.clip(rr, 1e-4, None))])
        pos, r, val = _optimize_once(params0, [300.0, 800.0], 400, deadline)
        if _is_valid(pos, r):
            score = r.sum()
            if score > best_score:
                best_score = score
                best_out = np.column_stack([pos, r])

    # final safety: if best_out invalid for any reason, fall back to grid
    if not _is_valid(best_out[:, 0], best_out[:, 1]):
        pos0, r0 = _initial_grid(N)
        best_out = np.column_stack([pos0, r0])

    return best_out
