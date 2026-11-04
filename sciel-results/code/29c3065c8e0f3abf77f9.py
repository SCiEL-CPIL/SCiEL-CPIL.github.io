"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 26
_T0 = time.time()

def _remaining(limit):
    return max(0.0, limit - (time.time() - _T0))


def _max_radius_for_positions(pos):
    dists = cdist(pos, pos)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1) / 2.0
    boundary = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                          np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(boundary, min_pair)
    return np.maximum(radii, 0.0)


def _is_valid(pos, radii=None, eps=1e-9):
    if radii is None:
        radii = _max_radius_for_positions(pos)
    if np.any(radii < -eps) or np.any(radii > 0.5 + eps):
        return False
    dists = cdist(pos, pos)
    if np.any(dists + eps < (radii[:, None] + radii[None, :])):
        return False
    if np.any(pos[:, 0] < radii - eps) or np.any(pos[:, 0] > 1.0 - radii + eps) or \
       np.any(pos[:, 1] < radii - eps) or np.any(pos[:, 1] > 1.0 - radii + eps):
        return False
    return True


def _soft_objective(v, n, beta):
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol)) / beta
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    viol = r[:, None] + r[None, :] - dist
    np.fill_diagonal(viol, -np.inf)
    pair_pen = np.sum(np.logaddexp(0.0, beta * viol)) / beta
    return -np.sum(r) + 50.0 * (wall_pen + pair_pen)


def _soft_gradient(v, n, beta):
    x = v[0:n]
    y = v[n:2 * n]
    r = v[2 * n:3 * n]
    grad = np.zeros(3 * n)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
    factor = 50.0 * sig
    for i in range(n):
        grad[2 * n + i] += factor[i]
        d = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        mi = int(np.argmin(d))
        if mi == 0:
            grad[i] -= factor[i]
        elif mi == 1:
            grad[i] += factor[i]
        elif mi == 2:
            grad[n + i] -= factor[i]
        else:
            grad[n + i] += factor[i]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, 1e-12)
    viol = r[:, None] + r[None, :] - dist
    np.fill_diagonal(viol, -np.inf)
    sig = 1.0 / (1.0 + np.exp(-beta * viol))
    np.fill_diagonal(sig, 0.0)
    factor = 50.0 * sig
    grad[2 * n:3 * n] += np.sum(factor, axis=1)
    grad[0:n] += np.sum(factor * dx / dist, axis=1)
    grad[n:2 * n] += np.sum(factor * dy / dist, axis=1)
    grad[2 * n:3 * n] -= 1.0
    return grad


def _gradient_polish(pos, deadline, beta=120.0, maxiter=400):
    n = len(pos)
    r_init = _max_radius_for_positions(pos)
    v0 = np.concatenate([pos[:, 0], pos[:, 1], r_init])
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    try:
        res = minimize(_soft_objective, v0, args=(n, beta),
                       jac=_soft_gradient, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-18,
                                'gtol': 1e-14, 'maxls': 60})
    except Exception:
        return pos, r_init, float(np.sum(r_init))
    if time.time() > deadline:
        return pos, r_init, float(np.sum(r_init))
    v_opt = res.x
    pos_opt = np.column_stack([v_opt[0:n], v_opt[n:2 * n]])
    pos_opt = np.clip(pos_opt, 0.0, 1.0)
    r_exact = _max_radius_for_positions(pos_opt)
    obj_opt = float(np.sum(r_exact))
    if _is_valid(pos_opt, r_exact, eps=1e-9):
        return pos_opt, r_exact, obj_opt
    return pos, r_init, float(np.sum(r_init))


def _objective(pos):
    return float(np.sum(_max_radius_for_positions(pos)))


def _hex_seed():
    rng = np.random.RandomState(0)
    pos = []
    for row in range(8):
        y = (row + 0.5) / 8.0
        offset = 0.0 if row % 2 == 0 else 0.5 / 8.0
        for col in range(8):
            x = (col + offset + 0.5) / 8.0
            pos.append([x, y])
            if len(pos) == N:
                return np.array(pos[:N])
    return np.array(pos[:N])


def _random_seed(seed):
    rng = np.random.RandomState(seed)
    pos = []
    for _ in range(4000):
        if len(pos) >= N:
            break
        x, y = rng.random(), rng.random()
        if all(np.hypot(x - p[0], y - p[1]) > 0.05 for p in pos):
            pos.append([x, y])
    while len(pos) < N:
        pos.append([rng.random(), rng.random()])
    return np.array(pos[:N])


def _simulated_annealing(pos, t_start, t_end, n_iter, seed, deadline):
    rng = np.random.RandomState(seed)
    cur = pos.copy()
    cur_obj = _objective(cur)
    best = cur.copy()
    best_obj = cur_obj
    n = len(cur)
    for it in range(n_iter):
        if time.time() > deadline:
            break
        t = t_start * (t_end / t_start) ** (it / n_iter)
        i = rng.randint(n)
        step = 0.05 * t + 0.001
        nx = np.clip(cur[i, 0] + rng.uniform(-step, step), 0.0, 1.0)
        ny = np.clip(cur[i, 1] + rng.uniform(-step, step), 0.0, 1.0)
        new = cur.copy()
        new[i] = [nx, ny]
        no = _objective(new)
        d = no - cur_obj
        if d > 0 or rng.random() < np.exp(d / t):
            cur = new
            cur_obj = no
            if cur_obj > best_obj:
                best = cur.copy()
                best_obj = cur_obj
    return best, best_obj


def _best_fit_local_search(pos, n_iter, seed, deadline):
    rng = np.random.RandomState(seed)
    cur = pos.copy()
    cur_obj = _objective(cur)
    n = len(cur)
    base_step = 0.012
    for it in range(n_iter):
        if time.time() > deadline:
            break
        improved = False
        step = base_step * max(0.2, 1.0 - it / (n_iter * 0.7))
        for i in rng.permutation(n):
            if time.time() > deadline:
                break
            xi, yi = cur[i]
            best_pos = None
            best_obj = cur_obj
            for di in range(-4, 5):
                for dj in range(-4, 5):
                    if di == 0 and dj == 0:
                        continue
                    cx = xi + di * step
                    cy = yi + dj * step
                    if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                        continue
                    new = cur.copy()
                    new[i] = [cx, cy]
                    no = _objective(new)
                    if no > best_obj + 1e-14:
                        best_obj = no
                        best_pos = new
            for _ in range(6):
                ang = rng.uniform(0, 2 * np.pi)
                mag = step * rng.uniform(0.5, 1.5)
                cx = np.clip(xi + mag * np.cos(ang), 0.0, 1.0)
                cy = np.clip(yi + mag * np.sin(ang), 0.0, 1.0)
                new = cur.copy()
                new[i] = [cx, cy]
                no = _objective(new)
                if no > best_obj + 1e-14:
                    best_obj = no
                    best_pos = new
            if best_pos is not None:
                cur = best_pos
                cur_obj = best_obj
                improved = True
        if not improved:
            base_step *= 0.6
            if base_step < 0.0003:
                break
    return cur, cur_obj


def construct():
    deadline = _T0 + 140.0
    inits = [_hex_seed()]
    for s in range(6):
        inits.append(_random_seed(s * 997 + 13))
    best_pos = None
    best_obj = -np.inf
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        sa_pos, sa_obj = _simulated_annealing(init, 0.4, 0.0003, 20000,
                                              idx * 7 + 1, deadline - 25)
        if sa_obj > best_obj:
            best_pos, best_obj = sa_pos, sa_obj
        if time.time() > deadline - 20:
            break
        bf_pos, bf_obj = _best_fit_local_search(sa_pos, 4000, idx * 13 + 2,
                                                deadline - 15)
        if bf_obj > best_obj:
            best_pos, best_obj = bf_pos, bf_obj
    if best_pos is None:
        pos = _hex_seed()
        r = _max_radius_for_positions(pos)
        return np.column_stack([pos, r])
    if time.time() < deadline - 25:
        bf, bfo = _best_fit_local_search(best_pos, 8000, 777, deadline - 20)
        if bfo > best_obj:
            best_pos, best_obj = bf, bfo
    if time.time() < deadline - 15:
        for beta in [60.0, 120.0, 250.0, 500.0]:
            if time.time() > deadline - 10:
                break
            p, r, o = _gradient_polish(best_pos, deadline - 8, beta=beta, maxiter=300)
            if o > best_obj:
                best_pos, best_obj = p, o
    if time.time() < deadline - 8:
        p, r, o = _gradient_polish(best_pos, deadline - 5, beta=600.0, maxiter=500)
        if o > best_obj:
            best_pos, best_obj = p, o
    if time.time() < deadline - 3:
        bf, bfo = _best_fit_local_search(best_pos, 3000, 555, deadline - 2)
        if bfo > best_obj:
            best_pos, best_obj = bf, bfo
    if time.time() < deadline - 0.3:
        rng = np.random.RandomState(99999)
        for _ in range(300):
            if time.time() > deadline - 0.05:
                break
            i = rng.randint(len(best_pos))
            new = best_pos.copy()
            new[i, 0] = np.clip(best_pos[i, 0] + rng.uniform(-0.0002, 0.0002), 0.0, 1.0)
            new[i, 1] = np.clip(best_pos[i, 1] + rng.uniform(-0.0002, 0.0002), 0.0, 1.0)
            nr = _max_radius_for_positions(new)
            no = float(np.sum(nr))
            if no > best_obj + 1e-13:
                best_pos, best_obj = new, no
    r_final = _max_radius_for_positions(best_pos)
    return np.column_stack([best_pos, r_final])
