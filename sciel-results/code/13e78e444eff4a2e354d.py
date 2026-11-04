"""Combined approach: multi-start scaffolding + smooth LSE + Riesz + spherical GD + high-precision polish."""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _normalize(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _covering_radius(points):
    """Exact covering radius (radians) via convex hull facet circumcenters."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(0)
        n_samples = 60000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))))

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(max_r)


def _facet_data(points):
    """Return (radii, centers, simplices) for triangular facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), [], []

    radii = []
    centers = []
    simplices = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        radii.append(r)
        centers.append(v)
        simplices.append(simplex)

    return np.array(radii), centers, simplices


def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Smooth surrogate: log-sum-exp of facet circumradii.
    Returns (value, gradient) with gradient in ambient R^3 (not projected).
    """
    points = _normalize(points)
    radii, centers, simplices = _facet_data(points)
    n = len(radii)
    if n == 0:
        return 0.0, np.zeros_like(points)

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)

    exps = np.exp(beta * (radii - m))
    lse = m + np.log(np.sum(exps)) / beta
    w = exps / np.sum(exps)

    grad = np.zeros_like(points)
    for i in range(n):
        wi = w[i]
        if wi < 1e-15:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(max(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += wi * g_tan

    return float(lse), grad


def _riesz_energy_grad(points, s=2.0):
    """
    Riesz s-energy (sum over pairs of angular distance^{-s}) and gradient (tangent-projected).
    Lower s (2.0) is softer and less prone to Tammes-like local minima.
    """
    points = _normalize(points)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)

    ang_powers = angles ** (-s - 1)
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)
    np.fill_diagonal(dots, 1.0)

    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)

    grad = weights @ points
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(angles[iu] ** (-s)))
    return energy, grad_tan


def _lbfgs_combined(points, deadline=None, beta=50.0, s=2.0, riesz_weight=0.05,
                    maxiter=400, ftol=1e-14, gtol=1e-12, maxfun=600):
    """
    L-BFGS-B on combined LSE covering radius + Riesz energy objective.
    The Riesz term acts as a smooth regularizer to avoid degenerate configurations.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        lse_val, lse_grad = _smooth_covering_radius_grad(x, beta=beta)
        riesz_val, riesz_grad = _riesz_energy_grad(x, s=s)
        val = lse_val + riesz_weight * riesz_val
        grad = lse_grad + riesz_weight * riesz_grad
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr


def _lbfgs_pure_lse(points, deadline=None, beta=50.0, maxiter=400,
                    ftol=1e-14, gtol=1e-12, maxfun=600):
    """
    L-BFGS-B on pure LSE surrogate (no Riesz mixing).
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr


def _spherical_gd(points, deadline=None, beta=30.0, max_iters=3000,
                  lr_init=0.005, lr_min=1e-10):
    """
    Spherical gradient descent with backtracking line search.
    Provides a different optimization trajectory than L-BFGS.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0
    max_no_improve = 200

    for it in range(max_iters):
        if deadline is not None and time.time() >= deadline:
            break

        val, grad = _smooth_covering_radius_grad(points, beta=beta)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad_normed = grad / gn

        accepted = False
        trial_lr = lr
        for trial in range(8):
            cand = points + trial_lr * grad_normed
            cand = _normalize(cand)
            cand_cr = _covering_radius(cand)
            if cand_cr < best_cr - 1e-15:
                points = cand
                best = cand.copy()
                best_cr = cand_cr
                lr = min(trial_lr * 1.05, 0.05)
                no_improve = 0
                accepted = True
                break
            trial_lr *= 0.7

        if not accepted:
            no_improve += 1
            lr = max(lr * 0.85, lr_min)
            if no_improve > max_no_improve:
                break

    return best, best_cr


def _lloyd_iteration(points, max_iters=200, tol=1e-10, deadline=None):
    """
    Spherical Lloyd: move each point toward the mean of circumcenters of incident facets.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for _ in range(max_iters):
        if deadline is not None and time.time() >= deadline:
            break

        try:
            hull = ConvexHull(points)
        except Exception:
            break

        new_points = points.copy()
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)

        for i in range(len(points)):
            if not point_to_facets[i]:
                continue
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                vn = np.linalg.norm(v)
                if vn < 1e-12:
                    continue
                v /= vn
                d = np.dot(v, a)
                if d < 0:
                    v = -v
                circumcenters.append(v)
            if not circumcenters:
                continue
            centroid = np.mean(circumcenters, axis=0)
            cn = np.linalg.norm(centroid)
            if cn > 1e-12:
                new_points[i] = centroid / cn

        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best = points.copy()
        if diff < tol:
            break

    return best, best_cr


def _simulated_annealing(points, deadline=None, seed=123):
    """
    Short SA to escape local minima before final polish.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    current = points.copy()
    current_cr = best_cr

    rng = np.random.RandomState(seed)
    T = 0.025
    cooling = 0.996
    T_min = 1e-5

    while T > T_min:
        if deadline is not None and time.time() >= deadline:
            break

        n_pert = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        cand = current.copy()
        for idx in idxs:
            noise = rng.randn(3) * T
            cand[idx] = current[idx] + noise
            cand[idx] /= np.linalg.norm(cand[idx])

        new_cr = _covering_radius(cand)
        delta = new_cr - current_cr
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current = cand
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best = cand.copy()

        T *= cooling

    return best, best_cr


def _multi_beta_cascade(points, deadline):
    """
    Multi-stage beta cascade: start with soft LSE for broad basin exploration,
    then progressively tighten for precision.
    """
    best = _normalize(points)
    best_cr = _covering_radius(best)

    stages = [
        # (beta, riesz_weight, maxiter, ftol, gtol, maxfun)
        (10.0, 0.10, 250, 1e-10, 1e-8, 300),
        (30.0, 0.05, 300, 1e-12, 1e-10, 400),
        (80.0, 0.02, 350, 1e-13, 1e-11, 450),
        (200.0, 0.01, 350, 1e-14, 1e-12, 500),
        (500.0, 0.00, 400, 1e-15, 1e-13, 550),
    ]

    for beta, rw, maxiter, ftol, gtol, maxfun in stages:
        if deadline is not None and time.time() >= deadline:
            break
        remaining = max(1.0, deadline - time.time())
        stage_time = min(15.0, remaining * 0.55)
        local_deadline = time.time() + stage_time

        if rw > 0.005:
            pts, cr = _lbfgs_combined(
                best, deadline=local_deadline, beta=beta, s=2.0,
                riesz_weight=rw, maxiter=maxiter, ftol=ftol, gtol=gtol, maxfun=maxfun
            )
        else:
            pts, cr = _lbfgs_pure_lse(
                best, deadline=local_deadline, beta=beta,
                maxiter=maxiter, ftol=ftol, gtol=gtol, maxfun=maxfun
            )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    return best, best_cr


# ---------------- initializers ----------------

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _icosahedron_based_points(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mn = np.linalg.norm(mid)
        if mn > 1e-10:
            mid /= mn
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    return pts[:n]


def _golden_spiral_points(n, seed=7):
    """Alternative structured initializer with slight perturbation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n)
    z = 1.0 - 2.0 * i / n
    theta = 2.399963229728653 * i + rng.randn(n) * 0.01
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts


# ---------------- main ----------------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Diverse initial configurations
    inits = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _icosahedron_based_points(N),
        _golden_spiral_points(N),
    ]

    best = _normalize(inits[0])
    best_cr = _covering_radius(best)

    # Phase 1: short Lloyd on each init to get into a reasonable basin
    for i, init in enumerate(inits):
        if time.time() >= deadline - 25.0:
            break
        lloyd_deadline = time.time() + min(4.0, max(0.5, deadline - time.time() - 25.0))
        pts, cr = _lloyd_iteration(
            _normalize(init),
            max_iters=150,
            tol=1e-10,
            deadline=lloyd_deadline,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 2: short SA from best to escape local minima
    if time.time() < deadline - 30.0:
        sa_deadline = time.time() + min(10.0, max(1.0, deadline - time.time() - 30.0))
        pts, cr = _simulated_annealing(
            best,
            deadline=sa_deadline,
            seed=123,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 3: multi-beta LSE+Riesz cascade (main refinement)
    if time.time() < deadline - 15.0:
        pts, cr = _multi_beta_cascade(
            best,
            deadline=deadline - 8.0,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 4: spherical gradient descent from current best (different optimizer path)
    if time.time() < deadline - 8.0:
        gd_deadline = time.time() + min(8.0, max(1.0, deadline - time.time() - 8.0))
        pts, cr = _spherical_gd(
            best,
            deadline=gd_deadline,
            beta=50.0,
            max_iters=2000,
            lr_init=0.004,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 5: final high-precision polish (very tight LSE, no Riesz)
    if time.time() < deadline - 3.0:
        polish_deadline = deadline - 1.0
        pts, cr = _lbfgs_pure_lse(
            best,
            deadline=polish_deadline,
            beta=1000.0,
            maxiter=500,
            ftol=1e-16,
            gtol=1e-15,
            maxfun=800,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Final normalization and sanity check
    best = _normalize(best)
    if not np.all(np.isfinite(best)):
        best = _fibonacci_points(N)

    return best
# EVOLVE-BLOCK-END
