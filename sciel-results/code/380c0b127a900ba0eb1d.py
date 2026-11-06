"""Fifty equal spheres in a unit cube.

Maximizes the common radius r = min( min_{i<j} d_ij/2, min_i min_k (1/2 - |c_ik|) )
using a smooth log-sum-exp (soft-min) surrogate with annealed temperature,
optimized by L-BFGS-B, with a strict wall-clock deadline guard.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 120.0          # hard evaluator limit (seconds)
SAFETY = 25.0             # return well before the limit
RNG_SEED = 20240607

# ---------------------------------------------------------------------------
# Soft-min (log-sum-exp) surrogate and its analytic gradient
# ---------------------------------------------------------------------------

def _softmin_value_and_grad(pool, tau):
    """Return (value, dS/dp) for S = -tau * log sum exp(-p/tau).

    pool : (M,) array of radius candidates (half-distances and wall clearances)
    dS/dp : (M,)  == -w where w = softmax(-p/tau)
    """
    x = -pool / tau
    # shift for numerical stability
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    w = ex / s                      # softmax weights, sum to 1
    val = -tau * (np.log(s) + (np.max(-pool / tau)))  # = -tau*log(sum exp(-p/tau))
    return val, -w                  # dS/dp = -w


def _build_pool_and_grads(centers, tau):
    """Pool of radius candidates + per-candidate gradient wrt flattened centers."""
    c = centers.reshape(-1, 3)
    n = c.shape[0]
    # pairwise half-distances
    diff = c[:, None, :] - c[None, :, :]            # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)             # (n, n)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    half_d = 0.5 * dist[iu]                         # (P,)
    # wall clearances: 1/2 - |c_ik|
    wall = 0.5 - np.abs(c)                          # (n, 3)
    wall = wall.ravel()                             # (3n,)

    pool = np.concatenate([half_d, wall])
    val, dSdp = _softmin_value_and_grad(pool, tau)

    # gradient of S wrt centers (flat)
    grad = np.zeros(n * 3)
    # pair part
    w_pair = dSdp[:len(half_d)]                     # (P,)
    # for each pair (i,j) with i<j, contribution to grad[i], grad[j]
    # diff[i,j] = c_i - c_j ; dp/dc_i = 0.5*(c_i-c_j)/d ; dp/dc_j = -0.5*(c_i-c_j)/d
    di, dj = iu
    dvec = diff[di, dj]                             # (P,3)
    dn = dist[di, dj][:, None]                      # (P,1)
    unit = dvec / dn                                # (P,3)
    # dS/dc_i += w_pair * 0.5 * unit ; dS/dc_j += w_pair * (-0.5)*unit
    g = np.zeros(n * 3)
    np.add.at(g, (di * 3) + np.arange(3)[:, None] if False else None, 0)  # placeholder no-op
    # do it explicitly but vectorized per axis
    for ax in range(3):
        gi = di * 3 + ax
        gj = dj * 3 + ax
        contrib = 0.5 * w_pair * unit[:, ax]
        np.add.at(g, gi, contrib)
        np.add.at(g, gj, -contrib)
    # wall part
    w_wall = dSdp[len(half_d):]                     # (3n,)
    # p = 0.5 - |c_ik| ; dp/dc_ik = -sign(c_ik)
    sign_c = np.sign(c).ravel()
    g_wall = w_wall * (-sign_c)
    g += g_wall
    return val, g


# ---------------------------------------------------------------------------
# Objective: minimize -softmin  (L-BFGS-B), box constraints = cube
# ---------------------------------------------------------------------------

def _make_obj(tau):
    def obj(flat, grad=None):
        centers = flat.reshape(-1, 3)
        val, g = _build_pool_and_grads(centers, tau)
        # L-BFGS-B minimizes; we want to maximize softmin -> minimize -softmin
        if grad is None:
            return -val, -g
        return -val, -g
    return obj


def _optimize(centers, tau_seq, start_time, deadline, safety, maxiter_total):
    c = centers.copy()
    best = c
    best_val = -np.inf
    LO = -0.5
    HI = 0.5
    bounds = [(LO, HI)] * (N * 3)
    iters_left = maxiter_total
    for tau in tau_seq:
        if time.time() > start_time + (deadline - safety):
            break
        obj = _make_obj(tau)
        try:
            res = minimize(obj, c.ravel(), jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max(20, iters_left // len(tau_seq)),
                                    'gtol': 1e-8, 'ftol': 1e-12})
            c = np.clip(res.x.reshape(-1, 3), LO, HI)
            val, _ = _build_pool_and_grads(c, tau)
            if val > best_val:
                best_val = val
                best = c.copy()
        except Exception:
            pass
        iters_left -= max(20, iters_left // len(tau_seq))
        if iters_left <= 0:
            break
    return best, best_val


def _true_radius(c):
    """Exact (nonsmooth) radius for scoring."""
    c = c.reshape(-1, 3)
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    r_pair = 0.5 * dist.min()
    r_wall = (0.5 - np.abs(c).max())
    return min(r_pair, r_wall)


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------

def _init_grid_jitter(rng, jitter=0.05):
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N].astype(float)
    grid += rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid, -0.49, 0.49)


def _init_bcc_like(rng):
    # body-centered cubic style: corners + face/center offsets, scaled to fit
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    pts = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                pts.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                # add a body-center offset point
                pts.append(((x + 0.0) / m - 0.5, (y + 0.0) / m - 0.5, (z + 0.0) / m - 0.5))
    pts = np.array(pts[:N])
    # scale to fit inside cube with margin
    pts = pts * 0.85
    pts += rng.uniform(-0.02, 0.02, size=pts.shape)
    return np.clip(pts, -0.49, 0.49)


def _init_random(rng):
    return rng.uniform(-0.45, 0.45, size=(N, 3))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    inits = [
        _init_grid_jitter(rng, jitter=0.04),
        _init_grid_jitter(rng, jitter=0.08),
        _init_bcc_like(rng),
        _init_random(rng),
        _init_random(rng),
        _init_grid_jitter(rng, jitter=0.12),
    ]

    # annealed temperature schedule: start smooth, end sharp
    tau_seq = [0.05, 0.03, 0.02, 0.012, 0.008, 0.005, 0.003, 0.002, 0.0012, 0.0008]

    best_centers = None
    best_true = -np.inf

    for k, init in enumerate(inits):
        if time.time() > t0 + (DEADLINE - SAFETY):
            break
        centers, val = _optimize(init, tau_seq, t0, DEADLINE, SAFETY, maxiter_total=400)
        tr = _true_radius(centers)
        if tr > best_true:
            best_true = tr
            best_centers = centers.copy()

    # final high-precision polish on the best
    if best_centers is not None and time.time() < t0 + (DEADLINE - SAFETY):
        fine_tau = [0.001, 0.0006, 0.00035, 0.0002, 0.00012]
        centers, val = _optimize(best_centers, fine_tau, t0, DEADLINE, SAFETY, maxiter_total=300)
        tr = _true_radius(centers)
        if tr > best_true:
            best_true = tr
            best_centers = centers.copy()

    # safety: ensure validity (finite, inside cube)
    if best_centers is None:
        best_centers = _init_grid_jitter(np.random.default_rng(0), jitter=0.0)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.49, neginf=-0.49)
    return best_centers
