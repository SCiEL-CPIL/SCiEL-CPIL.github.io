"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26
EPS = 1e-9
DEADLINE = 140.0  # safety margin before 150s kill

def _penalty_cost_and_grad(pos, rad, mu):
    """Smooth penalty for overlaps and boundary violations.
    Returns (cost, grad_pos (N,2), grad_rad (N,))."""
    n = len(rad)
    # pairwise overlaps: distance - (r_i + r_j) should be >= 0
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    rsum = rad[:, None] + rad[None, :]
    # soft penalty for each pair where d < rsum
    # use a smooth "max(0, rsum - d)" via softplus-like
    overlap = rsum - d  # positive means overlap
    # soft: log(1 + exp(k*overlap))/k
    k = 20.0
    # numerically stable softplus
    sp = np.where(overlap > 0, np.log1p(np.exp(k * overlap)) / k, 0.0)
    # zero diagonal
    np.fill_diagonal(sp, 0.0)
    cost = mu * np.sum(sp)

    # gradient w.r.t. positions and radii
    # d sp / d overlap = sigmoid(k*overlap) (for overlap>0, else ~0)
    sig = np.where(overlap > 0, 1.0 / (1.0 + np.exp(-k * overlap)), 0.0)
    np.fill_diagonal(sig, 0.0)
    # d overlap / d pos_i = -(dx_i/d, dy_i/d) for pair (i,j) where i is row
    # d overlap / d d = -1
    # grad_pos for i from pair (i,j): mu * sig[i,j] * (dx[i,j]/d[i,j], dy[i,j]/d[i,j])
    # Note: overlap = rsum - d, d overlap/d pos_i = -d d/d pos_i = -(-dx/d) = dx/d for row i? 
    # Let's be careful. d = ||pos_i - pos_j||. d/d pos_i = (pos_i - pos_j)/d = (dx, dy)/d.
    # overlap = rsum - d. d overlap / d pos_i = -d d/d pos_i = -(dx/d, dy/d).
    # d sp/d pos_i = sp'(overlap) * d overlap/d pos_i = sig * (-(dx/d), -(dy/d))
    # cost = mu * sum sp. grad_pos_i = mu * sum_j sig[i,j] * (-(dx[i,j]/d[i,j]), -(dy[i,j]/d[i,j]))
    inv_d = np.where(d > 1e-9, 1.0 / d, 0.0)
    np.fill_diagonal(inv_d, 0.0)
    gdx = sig * (-dx * inv_d)
    gdy = sig * (-dy * inv_d)
    grad_pos = mu * np.array([np.sum(gdx, axis=1), np.sum(gdy, axis=1)]).T
    # d overlap / d r_i = 1 (for each pair i,j, r_i contributes +1)
    grad_rad = mu * np.sum(sig, axis=1)

    # boundary: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # violation v = r - x, r - (1-x), etc. softpenalty
    def soft_bnd(v, k=20.0):
        return np.where(v > 0, np.log1p(np.exp(k * v)) / k, 0.0)
    def soft_bnd_sig(v, k=20.0):
        return np.where(v > 0, 1.0 / (1.0 + np.exp(-k * v)), 0.0)
    bnd = np.stack([rad - pos[:, 0], rad - (1.0 - pos[:, 0]),
                    rad - pos[:, 1], rad - (1.0 - pos[:, 1])], axis=1)
    bnd_cost = mu * np.sum(soft_bnd(bnd))
    cost += bnd_cost
    bnd_sig = soft_bnd_sig(bnd)
    # d bnd / d pos: for col0 (r-x): -1; col1 (r-(1-x)): +1; col2 (r-y): -1; col3: +1
    gp0 = mu * (-bnd_sig[:, 0] + bnd_sig[:, 1])
    gp1 = mu * (-bnd_sig[:, 2] + bnd_sig[:, 3])
    grad_pos[:, 0] += gp0
    grad_pos[:, 1] += gp1
    # d bnd / d r = 1 for all four
    grad_rad += mu * np.sum(bnd_sig, axis=1)

    return cost, grad_pos, grad_rad


def _optimize(pos0, rad0, mu0, mu_min, mu_decay, max_iter, deadline, lr=0.01):
    """L-BFGS-like via simple projected gradient / use scipy if available."""
    from scipy.optimize import minimize
    pos = pos0.copy()
    rad = rad0.copy()
    mu = mu0

    def fun(v):
        p = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        # objective: maximize sum r - penalty
        c, _, _ = _penalty_cost_and_grad(p, r, mu)
        return -np.sum(r) + c

    def jac(v):
        p = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        c, gp, gr = _penalty_cost_and_grad(p, r, mu)
        g = np.concatenate([-np.ones(N) + gr, np.zeros(2 * N)])
        g[2 * N:] = gr
        g[:2 * N] = gp.ravel()
        return g

    best_v = None
    best_val = -np.inf
    for it in range(max_iter):
        if time.time() > deadline:
            break
        v0 = np.concatenate([pos.ravel(), rad])
        res = minimize(fun, v0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': 30, 'gtol': 1e-6})
        v = res.x
        p = v[:2 * N].reshape(N, 2)
        r = v[2 * N:]
        # evaluate true value (sum of radii if valid)
        valid, val = _check_valid(p, r)
        if valid and val > best_val:
            best_val = val
            best_v = v.copy()
        # update
        pos = p
        rad = r
        # decay mu
        mu = max(mu_min, mu * mu_decay)

    if best_v is not None:
        p = best_v[:2 * N].reshape(N, 2)
        r = best_v[2 * N:]
        return p, r, best_val
    return pos, rad, -np.inf


def _check_valid(pos, rad):
    n = len(rad)
    if np.any(rad < -1e-9):
        return False, 0.0
    rad = np.maximum(rad, 0.0)
    if np.any(pos[:, 0] - rad < -1e-7) or np.any(1 - pos[:, 0] - rad < -1e-7) or \
       np.any(pos[:, 1] - rad < -1e-7) or np.any(1 - pos[:, 1] - rad < -1e-7):
        return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < rad[i] + rad[j] - 1e-7:
                return False, 0.0
    return True, float(np.sum(rad))


def _grid_init(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:n])


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_val = -np.inf
    best_result = None

    # baseline grid
    grid = _grid_init(N)
    pos = grid[:, :2]
    rad = grid[:, 2]
    valid, val = _check_valid(pos, rad)
    if valid and val > best_val:
        best_val = val
        best_result = np.column_stack([pos, rad])

    # try a few random inits + optimize
    rng = np.random.default_rng(42)
    n_starts = 8
    max_iter_per_start = 6

    for s in range(n_starts):
        if time.time() > deadline - 10:
            break
        # random positions, small radii
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        rad = np.full(N, 0.03)
        # shuffle to avoid symmetry
        idx = rng.permutation(N)
        pos = pos[idx]
        rad = rad[idx]
        p, r, val = _optimize(pos, rad, mu0=50.0, mu_min=1.0, mu_decay=0.7,
                              max_iter=max_iter_per_start, deadline=deadline - 5)
        valid, val = _check_valid(p, r)
        if valid and val > best_val:
            best_val = val
            best_result = np.column_stack([p, r])

    # also optimize from grid
    if time.time() < deadline - 10:
        p, r, val = _optimize(pos.copy(), rad.copy(), mu0=50.0, mu_min=1.0,
                              mu_decay=0.7, max_iter=8, deadline=deadline - 5)
        valid, val = _check_valid(p, r)
        if valid and val > best_val:
            best_val = val
            best_result = np.column_stack([p, r])

    if best_result is None:
        # fallback
        return grid
    return best_result
