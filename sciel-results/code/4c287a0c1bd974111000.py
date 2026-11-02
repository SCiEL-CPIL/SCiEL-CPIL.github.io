"""Optimized packing of 50 equal circles in a unit square.

Design family (kept): binary search on radius r with an L-BFGS feasibility
polish (violation objective).  New within-family ingredient: a *spectral*
surrogate.  We build a dynamic k-nearest-neighbor graph whose edge weights are
inversely proportional to squared distances, and add a term that maximizes the
smallest eigenvalue of the graph Laplacian.  The smallest eigenvalue of the
Laplacian is a smooth, global proxy for the minimum separation (it is bounded
by the smallest edge weight ~ 1/d_min^2), so it encourages every circle to
expand uniformly rather than letting one pair collapse.  It is combined with the
violation objective so the search still converges to a feasible packing.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

N = 50
KNN_K = 8          # neighbors per node in the dynamic graph
W_SCALE = 0.005    # scale for edge weight = W_SCALE / d^2
SPEC_W = 0.05      # weight of the spectral term (kept small: a proxy, not a target)
EPS = 1e-12


def _compute_radius(centers):
    """True maximum common radius for a configuration (double precision)."""
    wall_dists = np.min(np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )), axis=1)
    min_wall = np.min(wall_dists)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_wall)


def _laplacian_smallest_eig(centers, k=KNN_K):
    """Smallest eigenvalue of the (weighted) kNN graph Laplacian.

    Edge weight w_ij = W_SCALE / d_ij^2  ->  large when circles are close,
    so the smallest eigenvalue grows as the packing becomes more uniform /
    the minimum separation increases.  This is a smooth global surrogate.
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)

    w = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(d2[i])[:k]
        for j in idx:
            if d2[i, j] > 0:
                w[i, j] = W_SCALE / d2[i, j]
    w = 0.5 * (w + w.T)  # symmetrize

    row = np.arange(n)
    L = csr_matrix((w.ravel(), (row, row)))  # diagonal = degree
    L = L - csr_matrix(w)                     # L = D - W

    try:
        # smallest algebraic eigenvalue of the PSD Laplacian
        vals = eigsh(L, k=1, which='SA', return_eigenvectors=False)
        return float(np.min(vals))
    except Exception:
        return 0.0


def _violation_objective(x, r, use_spec):
    """Violation of radius r, optionally plus the spectral surrogate."""
    centers = x.reshape(N, 2)

    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    wall_cost = np.sum(wall_viol ** 2)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2 * r - pair_dists)
    pair_cost = np.sum(pair_viol ** 2)

    cost = pair_cost + wall_cost
    if use_spec:
        cost -= SPEC_W * _laplacian_smallest_eig(centers)
    return cost


def _violation_grad(x, r):
    """Gradient of the violation objective (spectral term handled numerically)."""
    centers = x.reshape(N, 2)
    grad = np.zeros_like(centers)

    # --- wall gradient ---
    wall_dists = np.column_stack((
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    wall_viol = np.maximum(0, r - wall_dists)
    sign_x = np.sign(centers[:, 0])
    sign_y = np.sign(centers[:, 1])
    grad[:, 0] += 2 * wall_viol[:, 0] * sign_x
    grad[:, 1] += 2 * wall_viol[:, 1] * sign_y

    # --- pair gradient ---
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol = np.maximum(0, 2 * r - pair_dists)

    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        viol = pair_viol[k]
        if viol == 0:
            continue
        d_ij = pair_dists[k]
        vec = centers[i] - centers[j]
        grad_vec = vec / d_ij
        grad[i] -= 2 * viol * grad_vec
        grad[j] += 2 * viol * grad_vec

    return grad.flatten()


def _construct_initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal seed: good starting geometry for dense packings."""
    s = 1.0 / 7.0
    pts = []
    for j in range(7):
        for i in range(7):
            x = -0.5 + s * (i + (0.5 if j % 2 else 0))
            y = -0.5 + s * j * 0.8660254
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts[:N])
    if len(pts) < N:
        pts = _construct_initial_grid()
    return pts


def _soft_min_polish(centers, budget):
    """Final high-precision polish: minimize the soft-min of all constraints."""
    beta = 200.0
    bounds = [(-0.5, 0.5)] * (2 * N)

    def _obj(x):
        c = x.reshape(N, 2)
        wall = np.min(np.column_stack((
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1])
        )), axis=1)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, k=1)
        all_vals = np.concatenate([dists[iu], wall])
        m = np.min(all_vals)
        sm = m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (all_vals - m))))
        return -sm

    try:
        res = opt.minimize(_obj, centers.flatten(), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
        r = _compute_radius(res.x.reshape(N, 2))
        if r > _compute_radius(centers):
            return res.x.reshape(N, 2).copy()
    except Exception:
        pass
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before the 150s kill

    best_centers = _construct_initial_grid()
    best_radius = _compute_radius(best_centers)
    hexc = _hex_init()
    if _compute_radius(hexc) > best_radius:
        best_centers = hexc
        best_radius = _compute_radius(hexc)

    r_low = 0.0
    r_high = 0.08
    current_centers = best_centers.copy()
    bounds = [(-0.5, 0.5)] * (2 * N)

    for step in range(18):
        if time.time() > deadline - 12.0:
            break
        r_mid = (r_low + r_high) / 2.0
        x0 = current_centers.flatten()
        use_spec = (step % 2 == 0)  # alternate spectral-guided / plain polish
        try:
            res = opt.minimize(
                _violation_objective, x0,
                jac=_violation_grad,
                args=(r_mid,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            x_res = res.x
            r_actual = _compute_radius(x_res.reshape(N, 2))
            if r_actual >= r_mid - 1e-6:
                r_low = r_mid
                if r_actual > best_radius:
                    best_radius = r_actual
                    best_centers = x_res.reshape(N, 2).copy()
                current_centers = x_res.reshape(N, 2).copy()
            else:
                r_high = r_mid
                current_centers = x_res.reshape(N, 2).copy()
        except Exception:
            r_high = r_mid

    # High-precision polish with remaining budget.
    if time.time() < deadline - 3.0:
        best_centers = _soft_min_polish(best_centers, deadline - time.time())
        best_radius = _compute_radius(best_centers)

    best_centers = np.clip(best_centers, -0.5, 0.5)
    if not np.all(np.isfinite(best_centers)):
        return _construct_initial_grid()
    return best_centers
