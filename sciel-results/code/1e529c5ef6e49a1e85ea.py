"""
Packing 50 equal circles in a unit circle.
Strategy:
1. Diverse initializations (Hex lattice, Ring, Random).
2. Binary search on radius with smooth feasibility (minimize violation potential).
3. Soft-min L-BFGS-B refinement with increasing sharpness.
4. High-precision force-directed polish with adaptive step size.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

# Pre-compute indices for pairwise distances
_I_IDX, _J_IDX = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_I_IDX)


def _compute_score(centers):
    """Compute the packing score for given centers."""
    if centers is None or not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0

    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    min_bound = float(np.min(1.0 - norms))

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))

    return min(min_pair / 2.0, min_bound)


def _get_hex_lattice_init(n):
    """Generate hexagonal lattice initialization."""
    best_pts = None
    best_score = -1.0

    for spacing in [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]:
        k = int(1.0 / spacing) + 5
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])

        mask = radius <= 0.98
        points = points[mask]
        radius = radius[mask]

        if len(points) >= n:
            idx = np.argsort(radius, kind="stable")[:n]
            pts = points[idx]
            s = _compute_score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is not None:
        return best_pts
    return _get_random_init(n, seed=42)


def _get_random_init(n, seed=42):
    """Generate random initialization inside unit disk."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = np.sqrt(rng.uniform(0, 0.92))
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)


def _get_ring_init(n):
    """Generate ring-based initialization."""
    centers = [[0.0, 0.0]]
    rings = [(6, 0.26), (12, 0.52), (12, 0.78), (19, 0.92)]
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
    return np.array(centers[:n])


def _violation_obj_and_grad(flat, r, beta):
    """
    Smooth violation potential for given radius r.
    Minimizes the soft-sum of violations:
      violation_i = max(0, required - actual) for each constraint
    Using soft approximation: (1/beta)*ln(1 + exp(beta*(required - actual)))
    """
    pts = flat.reshape(N, 2)

    # Boundary: need ||p_i|| <= 1 - r
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_violation = dists_origin - (1.0 - r)  # positive = violated

    # Pairwise: need ||p_i - p_j|| >= 2r
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    pair_dists = dist_mat[_I_IDX, _J_IDX]
    pair_violation = 2.0 * r - pair_dists  # positive = violated

    all_violations = np.concatenate([boundary_violation, pair_violation])

    # Soft plus: (1/beta)*ln(1 + exp(beta * x)) approximates max(0, x)
    # For stability, use logsumexp(0, beta*x) / beta
    y = beta * all_violations
    max_y = np.max(y)
    # logsumexp(0, y) = max_y + log(1 + sum(exp(y - max_y)))
    exp_shifted = np.exp(y - max_y)
    log_sum = max_y + np.log(1.0 + np.sum(exp_shifted))
    obj_val = log_sum / beta

    # Gradient
    # d/dx soft_plus(x) = sigmoid(beta*x) = exp(beta*x)/(1+exp(beta*x))
    sig_boundary = np.exp(beta * boundary_violation - max_y) / (1.0 + np.sum(exp_shifted[N:]))
    # Wait, need to be more careful. Let me recompute.

    # Actually, let me use a simpler approach:
    # soft_plus(x) = (1/beta)*ln(1 + exp(beta*x))
    # d/dx soft_plus(x) = 1/(1 + exp(-beta*x)) = sigmoid(beta*x)

    # Recompute with proper sigmoid
    sig_bnd = 1.0 / (1.0 + np.exp(-beta * boundary_violation))
    sig_pair = 1.0 / (1.0 + np.exp(-beta * pair_violation))

    # Recompute obj with log-sum-exp for numerical stability
    # obj = sum_i soft_plus(violation_i) = sum_i (1/beta)*ln(1+exp(beta*violation_i))
    # For large beta*x, soft_plus ~ x; for small beta*x, ~ (1/beta)*ln(2)
    
    # Numerically stable computation:
    y_b = beta * boundary_violation
    y_p = beta * pair_violation
    
    # soft_plus(x) = max(0,x) + (1/beta)*ln(1 + exp(-beta*max(0,x)))
    # This is stable when x > 0: soft_plus = x + (1/beta)*ln(1+exp(-beta*x)) ~ x
    # When x < 0: soft_plus = (1/beta)*ln(1+exp(beta*x))
    
    def soft_plus(x, b):
        return np.maximum(0, x) + (1.0 / b) * np.log1p(np.exp(-b * np.maximum(0, x))) + \
               np.where(x < 0, (1.0 / b) * np.log1p(np.exp(b * x)), 0.0)

    # Simpler: just use the sigmoid-based gradient and compute obj as sum of soft_plus
    # soft_plus(x) = (1/beta) * log(1 + exp(beta*x))
    # For numerical stability:
    # if beta*x > 50: soft_plus ~= x
    # else: compute directly
    
    def _soft_plus_val(x, b):
        bx = b * x
        return np.where(bx > 50, x, (1.0 / b) * np.log1p(np.exp(bx)))
    
    obj_val = np.sum(_soft_plus_val(boundary_violation, beta)) + np.sum(_soft_plus_val(pair_violation, beta))

    # Gradient w.r.t positions
    grad_pts = np.zeros_like(pts)

    # Boundary: violation_i = ||p_i|| - (1-r)
    # d(violation)/d(p_i) = p_i / ||p_i||
    safe_d = np.where(dists_origin < 1e-10, 1.0, dists_origin)
    boundary_grad = (sig_bnd / safe_d)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    # Pairwise: violation_k = 2r - ||p_i - p_j||
    # d(violation)/d(p_i) = -(p_i - p_j) / ||p_i - p_j||
    # d(violation)/d(p_j) = (p_i - p_j) / ||p_i - p_j||
    d_pairs = pair_dists
    safe_d_pairs = np.where(d_pairs < 1e-10, 1.0, d_pairs)
    vecs = pts[_I_IDX] - pts[_J_IDX]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    # grad w.r.t p_i: -sig_k * unit_vec
    # grad w.r.t p_j: +sig_k * unit_vec
    grad_contrib_i = -sig_pair[:, np.newaxis] * unit_vecs
    grad_contrib_j = sig_pair[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, _I_IDX, grad_contrib_i)
    np.add.at(grad_pts, _J_IDX, grad_contrib_j)

    grad_flat = grad_pts.flatten()
    return obj_val, grad_flat


def _soft_min_obj_and_grad(flat, beta):
    """
    Soft-min objective: minimize (1/beta)*ln(sum(exp(-beta*x_k)))
    where x_k are all critical distances (pairwise/2 and boundary distances).
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    pair_dists = dist_mat[_I_IDX, _J_IDX] / 2.0

    all_dists = np.concatenate([boundary_dists, pair_dists])

    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))

    if sum_exp <= 0:
        return 0.0, np.zeros_like(flat)

    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    # Boundary gradient
    w_boundary = weights[0:N]
    safe_d = np.where(dists_origin < 1e-10, 1.0, dists_origin)
    boundary_grad = (-w_boundary / safe_d)[:, np.newaxis] * pts
    grad_pts += boundary_grad

    # Pair gradient
    w_pairs = weights[N:]
    d_pairs = dist_mat[_I_IDX, _J_IDX]
    safe_d_pairs = np.where(d_pairs < 1e-10, 1.0, d_pairs)
    vecs = pts[_I_IDX] - pts[_J_IDX]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    grad_contrib_i = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_j = -0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, _I_IDX, grad_contrib_i)
    np.add.at(grad_pts, _J_IDX, grad_contrib_j)

    grad_flat = grad_pts.flatten()
    return obj_val, grad_flat


def _binary_search_radius(init_pts, start_time, r_lo=0.10, r_hi=0.135, tol=1e-7):
    """
    Binary search on radius: for a given r, minimize the violation potential.
    If the minimum violation is <= tol, the packing is feasible for r.
    """
    best_r = r_lo
    best_pts = init_pts.copy()

    while r_hi - r_lo > tol and time.time() - start_time < DEADLINE - 15.0:
        r_mid = (r_lo + r_hi) / 2.0

        # Optimize to minimize violation for radius r_mid
        best_obj = np.inf
        best_x = init_pts.flatten().copy()

        for beta in [50.0, 200.0, 1000.0]:
            if time.time() - start_time > DEADLINE - 15.0:
                break
            try:
                res = minimize(
                    _violation_obj_and_grad,
                    best_x,
                    args=(r_mid, beta),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    new_pts = res.x.reshape(N, 2)
                    # Project into disk
                    norms = np.linalg.norm(new_pts, axis=1)
                    mask = norms > 1.0
                    if np.any(mask):
                        new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
                    new_obj = _violation_obj_and_grad(new_pts.flatten(), r_mid, beta)[0]
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_x = res.x.copy()
            except Exception:
                pass

        # Check feasibility
        if best_obj < 1e-5:
            # Feasible, try larger radius
            r_lo = r_mid
            best_pts = best_x.reshape(N, 2).copy()
            best_r = r_mid
        else:
            r_hi = r_mid

    return best_r, best_pts


def _soft_min_optimize(init_pts, start_time):
    """Optimize using soft-min with increasing beta."""
    current = init_pts.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()

    betas = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

    for beta in betas:
        if time.time() - start_time > DEADLINE - 15.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_pts = res.x.reshape(N, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_pts.copy()
                    current = new_pts.copy()
        except Exception:
            pass

    return best_centers, best_score


def _force_directed_polish(pts, r_target, steps=200, start_time=None):
    """
    High-precision force-directed polish with adaptive step size.
    """
    pts = pts.copy()
    lr = 0.01
    min_lr = 1e-6

    for step in range(steps):
        if start_time is not None and time.time() > DEADLINE - 3.0:
            break

        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist_mat, np.inf)

        force = np.zeros_like(pts)

        d_pairs = dist_mat[_I_IDX, _J_IDX]
        mask_pair = (d_pairs < 2 * r_target) & (d_pairs > 1e-12)
        if np.any(mask_pair):
            ii = _I_IDX[mask_pair]
            jj = _J_IDX[mask_pair]
            d = d_pairs[mask_pair]
            vecs = pts[ii] - pts[jj]
            units = vecs / d[:, np.newaxis]
            strengths = (2 * r_target - d) * 0.5
            np.add.at(force, ii, units * strengths[:, np.newaxis])
            np.add.at(force, jj, -units * strengths[:, np.newaxis])

        mask_bnd = boundary_dists < r_target
        if np.any(mask_bnd):
            bi = np.where(mask_bnd)[0]
            safe_dists = np.where(dists_origin[bi] < 1e-10, 1.0, dists_origin[bi])
            unit_out = pts[bi] / safe_dists[:, np.newaxis]
            strengths = r_target - boundary_dists[bi]
            force[bi] -= unit_out * strengths[:, np.newaxis]

        # Adaptive step: reduce if force is large
        max_force = np.max(np.linalg.norm(force, axis=1))
        if max_force > 0.05:
            eff_lr = lr * 0.3
        elif max_force > 0.01:
            eff_lr = lr * 0.7
        else:
            eff_lr = lr

        pts += eff_lr * force

        # Project back
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0 - r_target * 0.999
        if np.any(mask):
            pts[mask] = pts[mask] * (1.0 - r_target * 0.999) / dists[mask, np.newaxis]

        # Decay learning rate
        lr *= 0.995
        if lr < min_lr:
            break

    return pts


def _high_precision_polish(pts, r_target, start_time):
    """Final polish with many small steps."""
    if time.time() - start_time > DEADLINE - 5.0:
        return pts

    # Use even smaller steps
    polished = _force_directed_polish(pts, r_target, steps=300, start_time=start_time)

    # Verify and potentially do one more pass with tighter target
    score = _compute_score(polished)
    if score > r_target:
        # Try to push further
        polished2 = _force_directed_polish(polished, score * 1.001, steps=100, start_time=start_time)
        score2 = _compute_score(polished2)
        if score2 > score:
            return polished2
        return polished
    return polished


def _optimize_single(init_pts, start_time):
    """Full optimization pipeline for one initialization."""
    current = init_pts.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()

    # Phase 1: Soft-min optimization
    if time.time() - start_time < DEADLINE - 30.0:
        optimized, score = _soft_min_optimize(current, start_time)
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
            current = optimized.copy()

    # Phase 2: Binary search on radius from current best
    if time.time() - start_time < DEADLINE - 20.0:
        r_est = best_score
        r_lo = max(0.10, r_est * 0.95)
        r_hi = min(0.135, r_est * 1.05)
        r_found, pts_found = _binary_search_radius(current, start_time, r_lo, r_hi, tol=1e-6)
        score_found = _compute_score(pts_found)
        if score_found > best_score:
            best_score = score_found
            best_centers = pts_found.copy()
            current = pts_found.copy()

    # Phase 3: High-precision polish
    if time.time() - start_time < DEADLINE - 5.0:
        polished = _high_precision_polish(best_centers, best_score, start_time)
        if np.all(np.isfinite(polished)):
            pol_score = _compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_centers = polished.copy()

    return best_centers, best_score


def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None

    # Diverse initializations
    initializations = [
        _get_hex_lattice_init(N),
        _get_ring_init(N),
        _get_random_init(N, seed=1),
        _get_random_init(N, seed=42),
        _get_random_init(N, seed=99),
        _get_random_init(N, seed=123),
    ]

    # Optimize each initialization
    for i, init in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 25.0:
            break

        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]

        optimized, score = _optimize_single(init, start_time)

        if score > best_score:
            best_score = score
            best_centers = optimized.copy()

        # If we've already found a very good solution, reduce remaining work
        if best_score > 0.125 and time.time() - start_time > DEADLINE - 40.0:
            break

    if best_centers is None:
        best_centers = initializations[0]
        best_score = _compute_score(best_centers)

    # Global refinement: another pass on the best solution
    if time.time() - start_time < DEADLINE - 15.0:
        refined, refined_score = _optimize_single(best_centers, start_time)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined.copy()

    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 3.0:
        final_polished = _high_precision_polish(best_centers, best_score, start_time)
        if np.all(np.isfinite(final_polished)):
            final_score = _compute_score(final_polished)
            if final_score > best_score:
                best_score = final_score
                best_centers = final_polished.copy()

    # Safety: ensure output is valid
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _get_hex_lattice_init(N)

    norms = np.linalg.norm(best_centers, axis=1)
    if np.any(norms > 1.0):
        best_centers = best_centers / norms[:, np.newaxis]

    return best_centers
