"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(pts):
    """Compute the packing score: min pairwise dist/2, min (1-|c_i|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_cont = 1.0 - np.max(norms)
    return min(min_pair, min_cont)


def _obj_and_grad(pts, r_target, soft_beta):
    """
    Soft-max constraint violation objective with vectorized gradient.
    f(pts) = soft_max([|c_i|-(1-r), 2r-d_ij])
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r_target - dists
    iu = np.triu_indices(n, k=1)
    pair_viol_flat = pair_viol[iu]

    all_viol = np.concatenate([cont_viol, pair_viol_flat])

    max_v = np.max(all_viol)
    if max_v > 50.0 / soft_beta:
        return max_v, np.zeros_like(pts)

    exp_v = np.exp(soft_beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / soft_beta
    weights = exp_v / sum_exp

    grad = np.zeros_like(pts)

    # Container gradient (vectorized)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs

    # Pairwise gradient (vectorized via np.add.at)
    iw = weights[n:]  # weights for pairwise violations
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]

        # For each pair (i,j): grad[i] -= w * (p_i - p_j)/d, grad[j] += w * (p_i - p_j)/d
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)

    return soft_max_val, grad


def _ring_init(n, r_guess):
    """Initialize points in concentric rings matching known optimal structure."""
    pts = np.zeros((n, 2))
    pts[0] = [0.0, 0.0]
    remaining = n - 1
    ring_radius = r_guess * 2.0
    placed = 1
    while placed < n:
        n_on_ring = max(1, int(2.0 * np.pi * ring_radius / (2.0 * r_guess)))
        n_on_ring = min(n_on_ring, remaining)
        angles = 2.0 * np.pi * np.arange(n_on_ring) / n_on_ring + np.pi / n_on_ring
        for k in range(n_on_ring):
            pts[placed] = [ring_radius * np.cos(angles[k]), ring_radius * np.sin(angles[k])]
            placed += 1
        remaining = n - placed
        ring_radius += 2.0 * r_guess
    return pts


def _hex_lattice_init(n, r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= n:
        # Sort by radius, take closest n
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:n]
        return pts[idx]
    else:
        return _ring_init(n, r_guess)


def _perturb(pts, r_guess, rng, magnitude=0.02):
    """Add small random perturbation to break symmetry."""
    perturbed = pts.copy()
    noise = rng.normal(0, magnitude, size=perturbed.shape)
    perturbed += noise
    # Keep inside container
    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
    for i in range(len(perturbed)):
        if norms[i] > 1.0 - r_guess:
            scale = (1.0 - r_guess) / norms[i]
            perturbed[i] *= scale
    return perturbed


def _optimize_for_radius(pts_init, r, soft_beta, max_iter=2000, tol=1e-10):
    """Minimize soft constraint violation for fixed r using L-BFGS-B."""
    n = pts_init.shape[0]

    def objective(x):
        pts = x.reshape(n, 2)
        val, grad = _obj_and_grad(pts, r, soft_beta)
        return val, grad.flatten()

    x0 = pts_init.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                               'maxls': 50})
    pts_opt = result.x.reshape(n, 2)
    return pts_opt, result.fun


def _adaptive_beta_optimize(pts_init, r, start_beta=20.0, end_beta=5000.0,
                            n_stages=4, max_iter_per_stage=1500, tol=1e-10):
    """Optimize with adaptive increasing beta across stages."""
    current_pts = pts_init.copy()
    current_val = np.inf
    for stage in range(n_stages):
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        max_iter = max_iter_per_stage if stage < n_stages - 1 else max_iter_per_stage * 2
        current_pts, current_val = _optimize_for_radius(current_pts, r, beta,
                                                         max_iter=max_iter, tol=tol)
    return current_pts, current_val


def construct():
    start_time = time.time()
    deadline = start_time + 125.0  # Leave 25s safety margin

    rng = np.random.RandomState(42)

    # Initial guess for radius
    r_best = 0.12
    best_pts = _ring_init(N, r_best)
    best_score = _score(best_pts)

    # Also try hex lattice init
    hex_pts = _hex_lattice_init(N, r_best)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_pts = hex_pts.copy()

    # Binary search on radius with multiple initializations
    r_low = 0.10
    r_high = 0.145

    # First, quickly find feasible r_high
    for _ in range(3):
        if time.time() > deadline - 60.0:
            break
        r_mid = (r_low + r_high) / 2.0
        pts_init = _ring_init(N, r_mid)

        found_feasible = False
        for beta in [20.0, 100.0, 500.0]:
            if time.time() > deadline - 15.0:
                break
            pts_opt, val = _optimize_for_radius(pts_init, r_mid, beta, max_iter=800)
            score = _score(pts_opt)
            if score >= r_mid - 1e-8:
                found_feasible = True
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    r_best = r_mid
                break

        if found_feasible:
            r_low = r_mid
        else:
            r_high = r_mid

        if r_high - r_low < 1e-5:
            break

    # Multi-start with symmetry breaking
    n_starts = 4
    for s in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        r_target = r_low
        if s == 0:
            pts_init = best_pts.copy()
        elif s == 1:
            pts_init = _hex_lattice_init(N, r_target)
        else:
            pts_init = _perturb(best_pts, r_target, rng, magnitude=0.03 * s)

        pts_opt, val = _adaptive_beta_optimize(pts_init, r_target,
                                                start_beta=20.0, end_beta=5000.0,
                                                n_stages=3, max_iter_per_stage=1200)
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()

    # Final high-precision polish
    if time.time() < deadline - 8.0:
        r_polish = r_low
        pts_init = best_pts.copy()
        for beta in [500.0, 2000.0, 10000.0]:
            if time.time() > deadline - 3.0:
                break
            pts_opt, val = _optimize_for_radius(pts_init, r_polish, beta,
                                                 max_iter=3000, tol=1e-12)
            score = _score(pts_opt)
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
            pts_init = pts_opt

    # Validate
    best_pts = np.array(best_pts, dtype=np.float64)
    if np.any(~np.isfinite(best_pts)) or len(best_pts) != N:
        # Fallback to hexagonal grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_pts

# EVOLVE-BLOCK-END
