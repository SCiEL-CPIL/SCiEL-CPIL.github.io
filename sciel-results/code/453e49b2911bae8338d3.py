import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_circumradii(pts, simplices):
    """Vectorized computation of circumradii for all facets.
    Returns array of radii (radians) and array of circumcenter unit vectors."""
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    n = np.cross(a - b, a - c)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm > 1e-10
    n = n / np.maximum(n_norm[:, None], 1e-10)
    centroids = (a + b + c) / 3.0
    signs = np.where(np.einsum('ij,ij->i', n, centroids) < 0, -1.0, 1.0)
    n = n * signs[:, None]
    dots = np.clip(np.einsum('ij,ij->i', n, a), -1.0, 1.0)
    radii = np.arccos(dots)
    radii[~mask] = 0.0
    return radii, n, mask

def _covering_radius_deg(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.degrees(np.max(radii)))
    except Exception:
        return 180.0

def _covering_radius_rad(pts):
    try:
        hull = ConvexHull(pts)
        radii, _, _ = _facet_circumradii(pts, hull.simplices)
        return float(np.max(radii))
    except Exception:
        return np.pi

def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _electrostatic_refine(pts, n_iters=500, lr=0.01):
    """Repulsion-based refinement: each point is pushed away by all others via Coulomb-like force."""
    pts = pts.copy()
    for it in range(n_iters):
        D = pts @ pts.T
        chord2 = 2.0 - 2.0 * D
        np.fill_diagonal(chord2, 1.0)
        W = 1.0 / chord2
        np.fill_diagonal(W, 0.0)
        row_sum = W.sum(axis=1)
        forces = pts * row_sum[:, None] - (W @ pts)
        forces = forces - np.einsum('ij,ij->i', forces, pts)[:, None] * pts
        forces /= (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
        pts = pts + lr * forces
        pts = _normalize(pts)
    return pts

def _smooth_surrogate(pts, beta=8.0):
    """Smooth surrogate: log-sum-exp of squared facet radii (soft-max).
    Returns (surrogate_value, softmax_weights, radii, simplices, circumcenters, mask)."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    radii, cc, mask = _facet_circumradii(pts, hull.simplices)
    if not np.any(mask):
        return None
    valid = mask
    r_valid = radii[valid]
    cc_valid = cc[valid]
    simp_valid = hull.simplices[valid]
    r2 = r_valid * r_valid
    m = np.max(r2)
    exps = np.exp((r2 - m) * beta)
    s = np.sum(exps)
    if s < 1e-300:
        weights = np.ones_like(exps) / len(exps)
    else:
        weights = exps / s
    surrogate = m + np.log(s) / beta
    return surrogate, weights, r_valid, cc_valid, simp_valid

def _surrogate_gradient(pts, weights, simplices, circumcenters, beta=8.0):
    """Gradient of smooth surrogate w.r.t. each point (in R^3, then projected to tangent)."""
    grad = np.zeros_like(pts)
    for k in range(len(simplices)):
        w = weights[k]
        if w < 1e-8:
            continue
        c = circumcenters[k]
        a = pts[simplices[k, 0]]
        cos_r = np.clip(np.dot(c, a), -1.0, 1.0)
        sin_r = np.sqrt(max(1.0 - cos_r * cos_r, 1e-12))
        for j in range(3):
            idx = simplices[k, j]
            v = pts[idx]
            d = c - v
            d_t = d - np.dot(d, v) * v
            d_t_norm = np.linalg.norm(d_t)
            if d_t_norm < 1e-10:
                continue
            d_t = d_t / d_t_norm
            r_k = np.arccos(cos_r)
            mag = 2.0 * r_k / sin_r * w
            grad[idx] -= mag * d_t
    grad = grad - np.einsum('ij,ij->i', grad, pts)[:, None] * pts
    return grad

def _polish_high_beta(pts, beta=40.0, n_iters=200, initial_step=0.003):
    """High-precision polish using higher beta for sharper focus on worst facets."""
    pts = pts.copy()
    best_local = _covering_radius_deg(pts)
    best_local_pts = pts.copy()
    step = initial_step
    for it in range(n_iters):
        res = _smooth_surrogate(pts, beta=beta)
        if res is None:
            break
        surrogate, weights, r_valid, cc_valid, simp_valid = res
        # Move vertices toward circumcenters with adaptive step
        for k in range(len(simp_valid)):
            w = weights[k]
            if w < 1e-7:
                continue
            c = cc_valid[k]
            for j in range(3):
                idx = simp_valid[k, j]
                pts[idx] = pts[idx] + step * (c - pts[idx])
        pts = _normalize(pts)
        if it % 10 == 0:
            rad = _covering_radius_deg(pts)
            if rad < best_local:
                best_local = rad
                best_local_pts = pts.copy()
            step *= 0.98
    return best_local_pts, best_local

def _local_search(pts, rng, n_trials=15, perturb_scale=0.008):
    """Local search with random perturbations followed by polish."""
    best_pts = pts.copy()
    best_rad = _covering_radius_deg(pts)
    for trial in range(n_trials):
        # Random perturbation
        noise = rng.randn(N, 3) * perturb_scale
        pts_pert = _normalize(pts + noise)
        # Quick Lloyd iterations
        step = 0.004
        for it in range(30):
            res = _smooth_surrogate(pts_pert, beta=12.0)
            if res is None:
                break
            _, weights, r_valid, cc_valid, simp_valid = res
            for k in range(len(simp_valid)):
                w = weights[k]
                if w < 1e-6:
                    continue
                c = cc_valid[k]
                for j in range(3):
                    idx = simp_valid[k, j]
                    pts_pert[idx] = pts_pert[idx] + step * (c - pts_pert[idx])
            pts_pert = _normalize(pts_pert)
        rad = _covering_radius_deg(pts_pert)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts_pert.copy()
    return best_pts, best_rad

def construct():
    start_time = time.time()
    max_time = 145.0

    def _time_left():
        return max_time - (time.time() - start_time)

    best_points = _fibonacci_spiral()
    best_rad = _covering_radius_deg(best_points)

    rng = np.random.RandomState(42)

    # Phase 0: Generate diverse initializations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.3))
    inits.append(_fibonacci_spiral(0.7))
    inits.append(_fibonacci_spiral(1.1))
    for seed in [42, 123, 789]:
        r = np.random.RandomState(seed)
        pts = r.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)
    # Electrostatic-refined initializations
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.0), n_iters=200, lr=0.005))
    inits.append(_electrostatic_refine(_fibonacci_spiral(0.5), n_iters=200, lr=0.005))

    # Evaluate and sort initializations
    init_list = []
    for init in inits:
        if _time_left() < 3.0:
            break
        rad = _covering_radius_deg(init)
        init_list.append((rad, init.copy()))
    init_list.sort(key=lambda x: x[0])

    # Phase 1: Refine top initializations with Lloyd + gradient
    top_inits = [x[1] for x in init_list[:4]]

    for init_idx, init_pts in enumerate(top_inits):
        if _time_left() < 15.0:
            break
        pts = init_pts.copy()
        cur_rad = _covering_radius_deg(pts)
        if cur_rad < best_rad:
            best_rad = cur_rad
            best_points = pts.copy()

        # Phase 1a: Lloyd iterations with decaying step
        step = 0.015
        for it in range(300):
            if _time_left() < 5.0:
                break
            res = _smooth_surrogate(pts, beta=8.0)
            if res is None:
                break
            _, weights, r_valid, cc_valid, simp_valid = res
            for k in range(len(simp_valid)):
                w = weights[k]
                if w < 1e-6:
                    continue
                c = cc_valid[k]
                for j in range(3):
                    idx = simp_valid[k, j]
                    pts[idx] = pts[idx] + step * (c - pts[idx])
            pts = _normalize(pts)
            if it % 15 == 0:
                rad = _covering_radius_deg(pts)
                if rad < best_rad:
                    best_rad = rad
                    best_points = pts.copy()
                step *= 0.992

        # Phase 1b: Gradient descent on smooth surrogate
        step = 0.008
        for it in range(150):
            if _time_left() < 5.0:
                break
            res = _smooth_surrogate(pts, beta=10.0)
            if res is None:
                break
            _, weights, r_valid, cc_valid, simp_valid = res
            grad = _surrogate_gradient(pts, weights, simp_valid, cc_valid, beta=10.0)
            pts = pts - step * grad
            pts = _normalize(pts)
            if it % 15 == 0:
                rad = _covering_radius_deg(pts)
                if rad < best_rad:
                    best_rad = rad
                    best_points = pts.copy()
                step *= 0.985

        # Phase 1c: High-beta polish
        if _time_left() > 8.0:
            pts_polished, rad_polished = _polish_high_beta(pts, beta=40.0, n_iters=150, initial_step=0.002)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = pts_polished.copy()
            else:
                # Keep the pre-polish version if it was better
                rad_pre = _covering_radius_deg(pts)
                if rad_pre < best_rad:
                    best_rad = rad_pre
                    best_points = pts.copy()

        # Phase 1d: Local search around current best
        if _time_left() > 10.0:
            pts_ls, rad_ls = _local_search(best_points, rng, n_trials=12, perturb_scale=0.006)
            if rad_ls < best_rad:
                best_rad = rad_ls
                best_points = pts_ls.copy()

    # Final global polish on the best configuration found
    if _time_left() > 5.0:
        pts_final, rad_final = _polish_high_beta(best_points, beta=60.0, n_iters=100, initial_step=0.001)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = pts_final.copy()

    return best_points
