"""Combined packing: ring-structured init + energy-based L-BFGS + binary-search radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# Best known packing radius (Graham et al. 1998)
R_KNOWN = 0.1258254895304


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    return min(r_pair, r_bound)


def _energy(P, r):
    """Smooth energy: sum of squared overlaps (circle-circle and circle-boundary)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    return 0.5 * (np.sum(overlaps_cc ** 2) + np.sum(overlaps_cb ** 2))


def _energy_grad(P, r):
    """Gradient of the energy function."""
    grad = np.zeros_like(P)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
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


def _ring_initialization(r_est=0.1258):
    """
    Initialize with concentric rings: 1 + 6 + 12 + 18 + 13 = 50.
    """
    centers = []
    centers.append([0.0, 0.0])

    # Ring 1: 6 circles
    ring1_r = 2.0 * r_est
    for i in range(6):
        angle = 2.0 * np.pi * i / 6
        centers.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle)])

    # Ring 2: 12 circles
    ring2_r = 2.0 * np.sqrt(3.0) * r_est
    for i in range(12):
        angle = 2.0 * np.pi * i / 12
        centers.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle)])

    # Ring 3: 18 circles
    ring3_r = 4.0 * r_est
    for i in range(18):
        angle = 2.0 * np.pi * i / 18
        centers.append([ring3_r * np.cos(angle), ring3_r * np.sin(angle)])

    # Ring 4: 13 circles
    ring4_r = 5.5 * r_est
    for i in range(13):
        angle = 2.0 * np.pi * i / 13 + np.pi / 13
        centers.append([ring4_r * np.cos(angle), ring4_r * np.sin(angle)])

    pts = np.array(centers[:N])
    norms = np.linalg.norm(pts, axis=1)
    for i in range(len(pts)):
        if norms[i] > 1.0 - r_est:
            pts[i] *= (1.0 - r_est) / norms[i]
    return pts


def _hex_lattice_initialization():
    """Hexagonal lattice fallback."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    best_pts = None
    best_score = -1.0
    while spacing > 0.05:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            order = np.argsort(radius[mask])
            pts = points[order[:N]]
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        spacing *= 0.97
    if best_pts is None:
        best_pts = np.random.rand(N, 2) * 0.8 - 0.4
    return best_pts


def _random_initialization(seed=42):
    """Random initialization projected into disk."""
    rng = np.random.default_rng(seed)
    rand_pts = rng.uniform(-1.0, 1.0, size=(N, 2))
    norms = np.linalg.norm(rand_pts, axis=1)
    for i in range(N):
        if norms[i] > 0.85:
            rand_pts[i] *= 0.85 / norms[i]
    return rand_pts


def _optimize_for_radius(P_init, r_target, time_limit=10.0):
    """
    Optimize positions to minimize energy for a fixed target radius.
    Returns (P_opt, energy_opt).
    """
    start = time.time()

    def objective(x):
        P = x.reshape(N, 2)
        return _energy(P, r_target)

    def grad(x):
        P = x.reshape(N, 2)
        return _energy_grad(P, r_target).flatten()

    try:
        res = minimize(objective, P_init.ravel(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 5000})
        P_opt = res.x.reshape(N, 2)
        e_opt = objective(P_opt.ravel())
        # Validate
        if not np.all(np.isfinite(P_opt)):
            return P_init, objective(P_init.ravel())
        return P_opt, e_opt
    except Exception:
        return P_init, objective(P_init.ravel())


def _local_refine_force(P, r, deadline):
    """
    Force-based local refinement: push apart overlapping circles and pull
    boundary-violating circles inward.
    """
    P_curr = P.copy()
    r_curr = r
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)

        forces = np.zeros_like(P_curr)

        overlap_cc = np.maximum(2.0 * r_curr - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)

        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]

        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break

        best_step = 0
        best_score = r_curr
        for factor in [0.05, 0.02, 0.01, 0.005, 0.002]:
            step = factor * norm_forces / (N * N + 1)
            P_new = P_curr + step * forces
            new_score = _compute_score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        if best_step > 0:
            P_curr = P_curr + best_step * forces
            r_curr = best_score
        else:
            break
    return P_curr, r_curr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # --- Phase 1: Generate diverse initializations ---
    inits = []
    inits.append(_ring_initialization(0.1258))
    inits.append(_ring_initialization(0.125))
    inits.append(_ring_initialization(0.1265))
    inits.append(_hex_lattice_initialization())
    for seed in [42, 123, 456, 789]:
        inits.append(_random_initialization(seed))

    best_init = max(inits, key=_compute_score)
    best_score = _compute_score(best_init)

    # --- Phase 2: For each good init, run binary search on radius ---
    current_best_P = best_init.copy()
    current_best_r = best_score

    for init_idx, P_init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break

        init_score = _compute_score(P_init)
        if init_score < current_best_r * 0.95:
            continue  # skip poor inits

        # Binary search for maximum feasible radius
        r_lo = init_score * 0.99
        r_hi = min(init_score * 1.05, 0.135)

        # First verify r_lo is feasible
        P_test, e_test = _optimize_for_radius(P_init, r_lo, time_limit=8.0)
        if e_test > 1e-8:
            r_lo = init_score * 0.95
            P_test, e_test = _optimize_for_radius(P_init, r_lo, time_limit=8.0)
            if e_test > 1e-8:
                continue

        # Binary search
        for _ in range(6):
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
                break
            r_mid = (r_lo + r_hi) / 2.0
            P_mid, e_mid = _optimize_for_radius(P_test, r_mid, time_limit=8.0)
            if e_mid < 1e-8:
                r_lo = r_mid
                P_test = P_mid
            else:
                r_hi = r_mid

        # After binary search, P_test should be at r_lo
        if r_lo > current_best_r:
            current_best_r = r_lo
            current_best_P = P_test.copy()

    # --- Phase 3: Local force-based refinement on the best packing ---
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        current_best_P, current_best_r = _local_refine_force(
            current_best_P, current_best_r, deadline)

    # --- Phase 4: Final high-precision L-BFGS polish at the found radius ---
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        r_target = current_best_r
        P_final, e_final = _optimize_for_radius(current_best_P, r_target,
                                               time_limit=15.0)
        if e_final < 1e-10:
            final_score = _compute_score(P_final)
            if final_score > current_best_r:
                current_best_P = P_final
                current_best_r = final_score

    # --- Final validation ---
    final_pts = current_best_P
    norms = np.linalg.norm(final_pts, axis=1)
    if not np.all(np.isfinite(final_pts)) or np.any(norms >= 1.0):
        # Fallback
        return best_init

    return final_pts
