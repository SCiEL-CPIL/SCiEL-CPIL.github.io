"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_H = np.sqrt(3.0) / 2.0
_VERT = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _H]])


def _bary_to_cart(b):
    # b has shape (n, 3), rows sum to 1
    return b @ _VERT


def _cart_to_bary(p):
    # p shape (n,2)
    x = p[:, 0]
    y = p[:, 1]
    c3 = y / _H
    c1 = 1.0 - x - c3
    c2 = x
    return np.column_stack([c1, c2, c3])


def _areas_cart(p):
    # p shape (n,2)
    n = p.shape[0]
    a = np.zeros((n, n, n), dtype=float)
    # vectorized
    for i in range(n):
        pi = p[i]
        # vectors from i to all
        v = p - pi
        cross = v[:, 0] * v[:, 1] - v[:, 1] * v[:, 0]  # zero
        # Actually compute for pairs j,k
        # Use cross product of (p_j - p_i) and (p_k - p_i)
        vj = p - pi  # (n,2)
        # cross_jk = (vj_j x vj_k) = vj_j0*vj_k1 - vj_j1*vj_k0
        cross = vj[:, 0][:, None] * vj[:, 1][None, :] - vj[:, 1][:, None] * vj[:, 0][None, :]
        # triangle area = 0.5 * |cross|
        a[i] = 0.5 * np.abs(cross)
    return a


def _min_area_cart(p):
    a = _areas_cart(p)
    # exclude degenerate same-index
    idx = ~np.eye(len(p), dtype=bool)
    # for each (i,j) consider k != i,j
    # simpler: brute
    n = len(p)
    mn = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = p[j] - p[i]
                v2 = p[k] - p[i]
                cr = v1[0] * v2[1] - v1[1] * v2[0]
                ar = 0.5 * abs(cr)
                if ar < mn:
                    mn = ar
    return mn


def _is_inside_bary(b, eps=1e-9):
    return np.all(b >= -eps)


def _objective(x, n):
    # x: (3n + 1) -> b (3n) and t
    b = x[:3 * n].reshape(n, 3)
    t = x[3 * n]
    p = _bary_to_cart(b)
    a = _min_area_cart(p)
    return -(a - t)


def _grad_objective(x, n):
    b = x[:3 * n].reshape(n, 3)
    t = x[3 * n]
    p = _bary_to_cart(b)
    a = _min_area_cart(p)
    # gradient of -(a - t) w.r.t. b and t
    # d/dt = 1
    # d/d b = - d a / d b ; approximate numerically
    g = np.zeros_like(x)
    g[3 * n] = 1.0
    h = 1e-7
    for idx in range(3 * n):
        xp = x.copy()
        xp[idx] += h
        bp = xp[:3 * n].reshape(n, 3)
        pp = _bary_to_cart(bp)
        ap = _min_area_cart(pp)
        g[idx] = -((ap) - a) / h
    return g


def _constraints_bary(x, n):
    b = x[:3 * n].reshape(n, 3)
    t = x[3 * n]
    # each bary sum to 1
    cons = []
    for i in range(n):
        cons.append({'type': 'eq', 'fun': lambda x, i=i: np.sum(x[3 * i:3 * i + 3]) - 1.0})
    # b >= 0 (use bary >= 0)
    for i in range(n):
        for j in range(3):
            idx = 3 * i + j
            cons.append({'type': 'ineq', 'fun': lambda x, idx=idx: x[idx]})
    # t <= min area: t - min_area <= 0 -> min_area - t >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: _min_area_cart(_bary_to_cart(x[:3 * n].reshape(n, 3))) - x[3 * n]})
    return cons


def construct():
    start = time.time()
    deadline = start + 145.0
    n = N
    H = _H

    def rand_bary(rng):
        # random point in barycentric simplex
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                return np.array([1.0 - u - v, u, v])

    def rand_points_bary(rng, k):
        return np.array([rand_bary(rng) for _ in range(k)])

    def score_points(p):
        return _min_area_cart(p)

    # initial random best
    rng = np.random.default_rng(7)
    best_b = None
    best_val = -np.inf
    best_p = None

    for seed in range(30):
        if time.time() > deadline - 5.0:
            break
        rng = np.random.default_rng(seed)
        b = rand_points_bary(rng, n)
        p = _bary_to_cart(b)
        val = score_points(p)
        if val > best_val:
            best_val = val
            best_b = b.copy()
            best_p = p.copy()

    # NLP local optimization from best random
    if best_b is not None:
        x0 = np.concatenate([best_b.flatten(), [best_val * 0.5]])
        bounds = [(0.0, 1.0)] * (3 * n) + [(0.0, 0.5)]
        cons = _constraints_bary(x0, n)
        try:
            res = minimize(_objective, x0, method='SLSQP', jac=_grad_objective,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if time.time() < deadline - 2.0:
                x_opt = res.x
                b_opt = x_opt[:3 * n].reshape(n, 3)
                # clamp to feasible
                b_opt = np.clip(b_opt, 0.0, 1.0)
                b_opt = b_opt / b_opt.sum(axis=1, keepdims=True)
                p_opt = _bary_to_cart(b_opt)
                val_opt = score_points(p_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_b = b_opt.copy()
                    best_p = p_opt.copy()
        except Exception:
            pass

    # additional random restarts with SLSQP if time permits
    for seed in range(30, 120):
        if time.time() > deadline - 10.0:
            break
        rng = np.random.default_rng(seed)
        b = rand_points_bary(rng, n)
        p = _bary_to_cart(b)
        val = score_points(p)
        if val >= best_val * 0.9:
            x0 = np.concatenate([b.flatten(), [val * 0.5]])
            bounds = [(0.0, 1.0)] * (3 * n) + [(0.0, 0.5)]
            cons = _constraints_bary(x0, n)
            try:
                res = minimize(_objective, x0, method='SLSQP', jac=_grad_objective,
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 100, 'ftol': 1e-12})
                x_opt = res.x
                b_opt = x_opt[:3 * n].reshape(n, 3)
                b_opt = np.clip(b_opt, 0.0, 1.0)
                b_opt = b_opt / b_opt.sum(axis=1, keepdims=True)
                p_opt = _bary_to_cart(b_opt)
                val_opt = score_points(p_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_b = b_opt.copy()
                    best_p = p_opt.copy()
            except Exception:
                pass

    if best_p is None:
        rng = np.random.default_rng(0)
        best_p = _bary_to_cart(rand_points_bary(rng, n))

    return best_p
# EVOLVE-BLOCK-END
