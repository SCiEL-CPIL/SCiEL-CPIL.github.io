"""Combined approach: hard-disk relaxation + soft-min L-BFGS-B + vectorized SLSQP epigraph polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")

N = 50
_DEADLINE = 140.0  # seconds; keep 10s safety margin under 150s kill


def _radius_fast(c):
    """Compute max common radius for centres c (N,2) in [-0.5,0.5]^2."""
    wall_r = np.min(0.5 - np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_r = d.min() / 2.0
    return min(wall_r, pair_r)


def _soft_min_radius(c, t):
    """
    Differentiable surrogate: soft-min of all pairwise half-distances
    and wall distances. As t -> 0 this approaches the true min radius.
    """
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    # Clip to avoid overflow
    exps = np.exp(-np.clip((all_terms - m) / t, 0, 700))
    soft = m - t * np.log(exps.sum())
    return soft


def _soft_min_radius_neg(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)


def _soft_min_radius_grad(x, t):
    """Vectorized gradient of negative soft-min radius w.r.t. x (flattened centres)."""
    c = x.reshape(N, 2)
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()

    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()

    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    exps = np.exp(-np.clip((all_terms - m) / t, 0, 700))
    Z = exps.sum()
    if Z == 0:
        return np.zeros(2 * N)

    weights = exps / Z
    grad_flat = np.zeros(2 * N)

    # Wall gradients: term = 0.5 - |c_ik|, d/dc_ik = -sign(c_ik)
    sign_c = np.sign(c)
    w_grad = -sign_c.ravel()
    grad_flat[:2 * N] += weights[:2 * N] * w_grad

    # Pair gradients: vectorized
    # For each pair (i,j), weight w_ij + w_ji contributes to both i and j
    # d term_ij / dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # d term_ij / dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    # Total contribution to c_i from all j: sum_j (w_ij + w_ji) * (c_i - c_j) / (2*d_ij)
    # But note: w_ij appears in term_ij (grad wrt c_i) and w_ji appears in term_ji (grad wrt c_i)
    # Actually term_ij and term_ji are the same distance, so:
    # grad_c_i = sum_{j!=i} (w_ij + w_ji) * (c_i - c_j) / (2 * d_ij)
    
    # Compute pair weights matrix: W[i,j] = w_ij + w_ji for i != j
    w_pair = np.zeros((N, N))
    # weights[2*N + i*N + j] is w_ij
    wp_flat = weights[2 * N:]
    wp_matrix = wp_flat.reshape(N, N)
    w_pair = wp_matrix + wp_matrix.T
    np.fill_diagonal(w_pair, 0.0)

    # diff[i,j] = c_i - c_j
    diff = c[:, None, :] - c[None, :, :]
    # dist matrix (already computed as d, but diagonal is inf)
    dist_safe = d.copy()
    np.fill_diagonal(dist_safe, 1.0)
    
    # grad_c_i = sum_j w_pair[i,j] * (c_i - c_j) / (2 * d_ij)
    # = sum_j (w_pair[i,j] / (2 * d_ij)) * (c_i - c_j)
    coeff = w_pair / (2.0 * dist_safe)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_flat = sum_j coeff[i,j] * diff[i,j]
    grad_pair = np.sum(coeff[:, :, None] * diff, axis=1)
    grad_flat[:2 * N] += grad_pair.ravel()

    return grad_flat


def _relax_hard_disk(c, R, iters, eta0, eta1):
    """
    Hard-disk relaxation: push overlapping circles apart and keep them
    inside the square. Vectorized.
    """
    c = c.copy()
    for it in range(iters):
        t = it / max(iters - 1, 1)
        eta = eta0 + (eta1 - eta0) * t
        diff = c[:, None, :] - c[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / np.maximum(d, 1e-12)
        np.fill_diagonal(mag, 0.0)
        f = np.sum(mag[:, :, None] * diff, axis=1)
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _slsqp_epigraph_polish(c, t0, time_budget=30.0):
    """
    Vectorized SLSQP epigraph: maximize r s.t. min_pair_dist >= 2r, min_wall_dist >= r.
    Uses only 2 constraint functions (vectorized internally) instead of 1250.
    """
    c = c.copy()
    r_init = _radius_fast(c)

    def _min_pair_dist(pts):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        return d.min()

    def _min_wall_dist(pts):
        return np.min(0.5 - np.abs(pts))

    def objective(x):
        return -x[-1]

    def constraint_pair(x):
        pts = x[:2 * N].reshape(N, 2)
        r = x[-1]
        return _min_pair_dist(pts) - 2.0 * r

    def constraint_wall(x):
        pts = x[:2 * N].reshape(N, 2)
        r = x[-1]
        return _min_wall_dist(pts) - r

    constraints = [
        {'type': 'ineq', 'fun': constraint_pair},
        {'type': 'ineq', 'fun': constraint_wall},
    ]

    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(1e-6, 0.5)]

    best_x = x0.copy()
    best_r = r_init

    maxiters_list = [200, 400, 800, 1500]
    for maxiter in maxiters_list:
        if (time.time() - t0) > time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'iprint': 0}
            )
            if res.success or (res.x[-1] > best_r - 1e-10):
                c_new = res.x[:2 * N].reshape(N, 2)
                r_verified = _radius_fast(c_new)
                if r_verified > best_r:
                    best_r = r_verified
                    best_x = res.x.copy()
                    x0 = res.x.copy()
        except Exception:
            pass

    return best_x[:2 * N].reshape(N, 2), best_r


def _grid_init(m, n, seed=0):
    """Create an m x n grid of points in the unit square."""
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n) + 0.5) / n - 0.5
    pts = np.array([(x, y) for y in ys for x in xs])
    if len(pts) > N:
        idx = np.linspace(0, len(pts) - 1, N, dtype=int)
        pts = pts[idx]
    return pts


def _random_init(seed=0):
    """Random initialization with jitter around grid."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    base = _grid_init(m, m, seed)
    base = base[:N].copy()
    jitter = rng.uniform(-0.02, 0.02, size=base.shape)
    base += jitter
    base = np.clip(base, -0.5, 0.5)
    return base


def _lbfgsb_soft_min(c, t0, time_budget=20.0):
    """
    Multi-temperature L-BFGS-B soft-min polish with vectorized gradient.
    """
    c = c.copy()
    best_r = _radius_fast(c)
    temps = [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]

    for t in temps:
        if time.time() - t0 > time_budget:
            break
        try:
            res = minimize(
                _soft_min_radius_neg,
                c.ravel(),
                jac=lambda x: -_soft_min_radius_grad(x, t),
                args=(t,),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_new = res.x.reshape(N, 2)
            r_new = _radius_fast(c_new)
            if r_new > best_r:
                best_r = r_new
                c = c_new
        except Exception:
            pass

    return c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    best_pts = None
    best_radius = 0.0
    candidates = []  # list of (radius, pts) after relaxation

    # Generate diverse initializations
    inits = []

    # 1. Standard grid
    m = int(np.ceil(np.sqrt(N)))
    inits.append(_grid_init(m, m, seed=0))

    # 2. Grid with different aspect ratios
    for mm, nn in [(7, 8), (8, 7), (6, 9), (9, 6), (5, 10), (10, 5)]:
        if mm * nn >= N:
            inits.append(_grid_init(mm, nn, seed=1))

    # 3. Random inits
    for seed in range(5):
        inits.append(_random_init(seed=seed))

    # 4. Perturbed grid
    base_grid = _grid_init(m, m, seed=0)[:N].copy()
    for seed in range(3):
        rng = np.random.RandomState(seed + 100)
        perturbed = base_grid + rng.uniform(-0.03, 0.03, size=base_grid.shape)
        perturbed = np.clip(perturbed, -0.5, 0.5)
        inits.append(perturbed)

    # Phase 1+2: For each init, do relaxation + soft-min, collect top candidates
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break

        r0 = _radius_fast(init)
        if r0 > best_radius:
            best_radius = r0
            best_pts = init.copy()

        # Hard-disk relaxation
        if time.time() > deadline - 35:
            break
        r_relax = max(r0, 0.05)
        c_relax = _relax_hard_disk(init, r_relax, iters=80, eta0=0.5, eta1=0.1)
        r_relax = _radius_fast(c_relax)
        if r_relax > best_radius:
            best_radius = r_relax
            best_pts = c_relax.copy()

        # Soft-min polish
        if time.time() > deadline - 30:
            break
        c_soft, r_soft = _lbfgsb_soft_min(c_relax, time.time(), time_budget=5.0)
        if r_soft > best_radius:
            best_radius = r_soft
            best_pts = c_soft.copy()

        candidates.append((r_soft, c_soft))

    # Keep top 3 candidates for SLSQP polish
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:3]

    # Final phase: SLSQP epigraph polish on top candidates
    remaining_time = deadline - time.time() - 5
    if top_candidates and remaining_time > 10:
        # Allocate time: 60% to best, 25% to second, 15% to third
        time_allocs = [remaining_time * 0.6, remaining_time * 0.25, remaining_time * 0.15]
        for idx, (r_cand, c_cand) in enumerate(top_candidates):
            if time.time() > deadline - 5:
                break
            t_alloc = time_allocs[idx]
            try:
                c_final, r_final = _slsqp_epigraph_polish(c_cand, time.time(), time_budget=t_alloc)
                if r_final > best_radius:
                    best_radius = r_final
                    best_pts = c_final.copy()
            except Exception:
                pass

    if best_pts is None:
        best_pts = _grid_init(m, m, seed=0)[:N]

    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts
