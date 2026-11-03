"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    return min(r_pair, r_bound)

def _check_feasible(P, r, tol=1e-8):
    """Check if packing with radius r is feasible."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    max_origin = np.max(np.linalg.norm(P, axis=1))
    return (min_pair >= 2.0 * r - tol) and (max_origin + r <= 1.0 + tol)

# ---- squared-overlap energy (smooth, gradient available) ----
def _energy(P, r):
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    grad = np.zeros_like(P)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    W = V_cc * inv_dist
    sum_W = np.sum(W, axis=1)
    term1 = (P.T @ W).T
    term2 = P * sum_W[:, np.newaxis]
    grad += term1 - term2
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    return grad

# ---- log-sum-exp smooth energy (high-precision polish) ----
def _smooth_energy(P, r, beta=50.0):
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    violations_cc = 2.0 * r - dist
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0

    def _lse(v, b):
        vs = b * v
        vm = np.max(vs)
        if not np.isfinite(vm) or vm == -np.inf:
            return 0.0
        return (1.0 / b) * (vm + np.log(np.sum(np.exp(vs - vm))))

    return _lse(violations_cc, beta) + _lse(violations_cb, beta)

def _smooth_energy_grad(P, r, beta=50.0):
    grad = np.zeros_like(P)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    violations_cc = 2.0 * r - dist
    vs_cc = beta * violations_cc
    vm_cc = np.max(vs_cc)
    if np.isfinite(vm_cc) and vm_cc != -np.inf:
        w_cc = np.exp(vs_cc - vm_cc)
        w_cc /= np.sum(w_cc)
        inv_dist = np.zeros_like(dist)
        np.divide(1.0, dist, out=inv_dist, where=dist > 0)
        W_cc = w_cc * inv_dist
        sum_W = np.sum(W_cc, axis=1)
        term1 = (P.T @ W_cc).T
        term2 = P * sum_W[:, np.newaxis]
        grad += term2 - term1
    dist_origin = np.linalg.norm(P, axis=1)
    violations_cb = dist_origin + r - 1.0
    vs_cb = beta * violations_cb
    vm_cb = np.max(vs_cb)
    if np.isfinite(vm_cb) and vm_cb != -np.inf:
        w_cb = np.exp(vs_cb - vm_cb)
        w_cb /= np.sum(w_cb)
        inv_dist_origin = np.zeros_like(dist_origin)
        np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
        grad += (w_cb * inv_dist_origin)[:, np.newaxis] * P
    return grad

# ---- ring-structure generator (fixed topology, continuous radii/phases) ----
def _get_ring_points(params, ring_counts):
    k = len(ring_counts)
    P = np.zeros((N, 2))
    radii = params[:k]
    offsets = params[k:2 * k]
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count + n_pts, 0] = R * np.cos(angles)
            P[count:count + n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P

def _get_hex_initial_points():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _optimize_positions(P_init, r_target, time_limit=5.0, use_smooth=False, beta=50.0, max_iter=500):
    """Local refinement of P for a fixed r_target using L-BFGS-B."""
    start = time.time()
    if use_smooth:
        obj = lambda pf: _smooth_energy(pf.reshape(N, 2), r_target, beta)
        grd = lambda pf: _smooth_energy_grad(pf.reshape(N, 2), r_target, beta).flatten()
    else:
        obj = lambda pf: _energy(pf.reshape(N, 2), r_target)
        grd = lambda pf: _energy_grad(pf.reshape(N, 2), r_target).flatten()

    def _callback(state):
        return time.time() - start > time_limit

    try:
        res = minimize(obj, P_init.flatten(), jac=grd, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10,
                                'maxls': 30, 'callback': _callback})
        P_opt = res.x.reshape(N, 2)
        if not np.all(np.isfinite(P_opt)):
            P_opt = P_init
        return P_opt
    except Exception:
        return P_init

def _global_search_ring(ring_counts, time_limit=15.0, seed=42):
    """Differential evolution over ring radii + phases for a fixed topology."""
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k

    def neg_score(params):
        P = _get_ring_points(params, ring_counts)
        return -_compute_score(P)

    start = time.time()

    def _callback(xk, convergence):
        return time.time() - start > time_limit

    try:
        res = differential_evolution(neg_score, bounds, maxiter=40, popsize=14,
                                     tol=1e-5, seed=seed, polish=True,
                                     init='latinhypercube', callback=_callback)
        best_params = res.x
        return _get_ring_points(best_params, ring_counts)
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

    np.random.seed(42)

    # Baseline: hexagonal lattice
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)

    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 12, 17, 14],
    ]

    # Phase 1: global search for ring topology parameters
    for idx, ring_counts in enumerate(ring_structures):
        if time.time() > deadline - 40.0:
            break
        P_cand = _global_search_ring(ring_counts, time_limit=12.0, seed=42 + idx)
        if P_cand is None:
            continue
        s = _compute_score(P_cand)
        if s > score_best:
            P_best = P_cand
            score_best = s

    # Phase 2: binary search on r with smooth energy polish
    r_low = score_best
    r_high = 0.135
    for _ in range(9):
        if time.time() > deadline - 12.0:
            break
        r_mid = (r_low + r_high) / 2.0
        best_P = None
        best_E = np.inf
        for restart in range(3):
            if time.time() > deadline - 12.0:
                break
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = P_best + 0.003 * (restart + 1) * np.random.randn(N, 2)
            P_opt = _optimize_positions(P_init, r_mid, time_limit=2.5, use_smooth=True, beta=50.0)
            E = _smooth_energy(P_opt, r_mid, 50.0)
            if E < best_E:
                best_E = E
                best_P = P_opt
        if best_P is None:
            break
        if _check_feasible(best_P, r_mid):
            r_low = r_mid
            s = _compute_score(best_P)
            if s > score_best:
                P_best = best_P
                score_best = s
        else:
            r_high = r_mid

    # Phase 3: final high-precision polish
    if time.time() < deadline - 5.0:
        r_target = score_best
        for restart in range(4):
            if time.time() > deadline - 3.0:
                break
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = P_best + 0.001 * restart * np.random.randn(N, 2)
            P_opt = _optimize_positions(P_init, r_target, time_limit=1.5, use_smooth=True,
                                        beta=100.0, max_iter=500)
            s = _compute_score(P_opt)
            if s > score_best:
                P_best = P_opt
                score_best = s

    # Ensure all points strictly inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0 - 1e-10
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-10) / norms[mask]

    return P_best
# EVOLVE-BLOCK-END
