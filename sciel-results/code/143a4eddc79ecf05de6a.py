"""Sphere covering with 50 points - optimized algorithm."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians via convex-hull facet circumcenters."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 50000
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

def _fast_facets(points):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    out = []
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
        out.append((r, v, simplex))
    return out

def _riesz_energy_grad(points, s=12.0):
    """Compute Riesz s-energy and its gradient w.r.t. points on sphere."""
    points = np.asarray(points, dtype=float)
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
    
    iu = np.triu_indices(n, k=1)
    energy = np.sum(angles[iu] ** (-s))
    
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    return energy, grad_tan

def _lbfgs_riesz(points, max_time=20.0, start_time=None, s=12.0, maxiter=300):
    """Minimize Riesz s-energy via L-BFGS-B on the sphere."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norms[x_norms < 1e-15] = 1.0
        x = x / x_norms
        val, grad = _riesz_energy_grad(x, s=s)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)

        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius and its gradient w.r.t. points.
    Uses log-sum-exp over facet circumradii with proper gradient.
    Returns (value, gradient) where gradient has shape (N, 3).
    """
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 30000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        ang_dists = 2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))
        m = np.max(ang_dists)
        if m < 1e-12:
            return 0.0, np.zeros_like(points)
        val = m + np.log(np.sum(np.exp(beta * (ang_dists - m)))) / beta
        return val, np.zeros_like(points)

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

    if not radii:
        return 0.0, np.zeros_like(points)
    radii = np.array(radii)
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w /= w.sum()

    grad = np.zeros_like(points)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan

    return val, grad

def _lbfgs_refine(points, max_time=20.0, start_time=None, beta=50.0):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate for the covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norms[x_norms < 1e-15] = 1.0
        x = x / x_norms
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)

        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _gram_max_violation(points, beta=20.0):
    """
    Compute a smooth approximation of the maximum pairwise dot product (min separation).
    The maximum off-diagonal dot product is the violation of the separation constraint.
    We minimize this to maximize the minimum pairwise angular distance.
    Returns (value, gradient) where value is the soft-max of off-diagonal dots.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    dots = points @ points.T
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    off_diag = dots[mask]
    
    if len(off_diag) == 0:
        return 0.0, np.zeros_like(points)
    
    m = np.max(off_diag)
    # Soft max: m + (1/beta) * log(sum(exp(beta * (x - m))))
    exp_terms = np.exp(beta * (off_diag - m))
    val = m + np.log(np.sum(exp_terms)) / beta
    
    # Gradient of soft-max w.r.t. dots
    w = exp_terms / np.sum(exp_terms)
    # Gradient w.r.t. dots matrix
    grad_dots = np.zeros((n, n))
    idx = np.where(mask)
    grad_dots[idx] = w
    
    # Symmetrize (since dots is symmetric, and we want gradient w.r.t. points)
    grad_dots = (grad_dots + grad_dots.T) / 2.0
    # Subtract diagonal (should be zero anyway, but for safety)
    np.fill_diagonal(grad_dots, 0.0)
    
    # Gradient w.r.t. points: grad_points = grad_dots @ points
    grad = grad_dots @ points
    # Project to tangent space
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points
    
    return val, grad_tan

def _lbfgs_gram(points, max_time=20.0, start_time=None, beta=20.0):
    """
    Refine points to maximize minimum pairwise separation by minimizing
    the soft-max of pairwise dot products (Gram matrix approach).
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norms[x_norms < 1e-15] = 1.0
        x = x / x_norms
        val, grad = _gram_max_violation(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 300}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)

        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _grad_covering_radius(points):
    """Compute a gradient of the covering radius w.r.t. the points."""
    facets = _fast_facets(points)
    if not facets:
        return np.zeros_like(points), 0.0
    facets.sort(key=lambda x: -x[0])
    max_r = facets[0][0]
    K = min(10, len(facets))
    radii = np.array([f[0] for f in facets[:K]])
    beta = 30.0
    w = np.exp(beta * (radii - max_r))
    w /= w.sum()
    grad = np.zeros_like(points)
    for i in range(K):
        r, center, simplex = facets[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan
    return grad, max_r

def _spherical_gd(points, max_iters=2000, lr_init=0.008, lr_min=1e-9,
                  time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive step size using covering radius gradient."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        grad, cur_cr = _grad_covering_radius(points)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad /= gn
        accepted = False
        for trial in range(5):
            cand = points + lr * grad
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cand_cr = _covering_radius(cand)
            if cand_cr < best_cr - 1e-14:
                points = cand
                best = cand.copy()
                best_cr = cand_cr
                lr = min(lr * 1.05, 0.05)
                no_improve = 0
                accepted = True
                break
            lr = lr * 0.7
        if not accepted:
            no_improve += 1
            lr = max(lr * 0.85, lr_min)
            if no_improve > 80 or lr < lr_min * 1.01:
                break
    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """Lloyd-style: each point moves to the average circumcenter of its adjacent facets."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        facets = _fast_facets(points)
        if not facets:
            break
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended /= np.linalg.norm(blended, axis=1, keepdims=True)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended /= np.linalg.norm(blended, axis=1, keepdims=True)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42,
                         T_init=0.06, T_min=1e-6, cooling=0.997):
    """Simulated annealing with random perturbations."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(seed)
    T = T_init
    cur = points.copy()
    cur_cr = best_cr

    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        n_pert = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        cand = cur.copy()
        for i in idxs:
            noise = rng.randn(3) * T
            cand[i] = cur[i] + noise
            cand[i] /= np.linalg.norm(cand[i])
        cand_cr = _covering_radius(cand)
        delta = cand_cr - cur_cr
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-12)):
            cur = cand
            cur_cr = cand_cr
            if cand_cr < best_cr:
                best = cand.copy()
                best_cr = cand_cr
        T *= cooling
    return best, best_cr

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

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

def _jittered_fibonacci(n, seed=42, jitter=0.05):
    """Fibonacci sphere with small random jitter to break symmetry."""
    pts = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * jitter
    pts = pts + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _refine(points, time_limit=None, start_time=None):
    """
    Multi-stage refinement pipeline:
    1. Riesz L-BFGS (spread points)
    2. Gram max-violation L-BFGS (maximize min pairwise separation)
    3. Lloyd (local facet-based optimization)
    4. L-BFGS covering radius (smooth surrogate)
    5. Lloyd again
    6. L-BFGS covering radius (higher beta)
    7. Spherical GD (final polish)
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, _covering_radius(points)
        t1 = min(rem * 0.25, 20.0)
        t2 = min(rem * 0.15, 15.0)
        t3 = min(rem * 0.15, 10.0)
        t4 = min(rem * 0.20, 20.0)
        t5 = min(rem * 0.10, 8.0)
        t6 = min(rem * 0.10, 15.0)
        t7 = max(rem * 0.05, 2.0)
    else:
        t1 = 20.0
        t2 = 15.0
        t3 = 10.0
        t4 = 20.0
        t5 = 8.0
        t6 = 15.0
        t7 = 5.0

    # Stage 1: Riesz energy minimization
    points, cr = _lbfgs_riesz(points, max_time=t1, start_time=start_time, s=12.0)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr

    # Stage 2: Gram max-violation minimization
    points, cr = _lbfgs_gram(points, max_time=t2, start_time=start_time, beta=20.0)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr

    # Stage 3: Lloyd
    points, cr = _spherical_lloyd(points, max_iters=80, time_limit=t3, start_time=start_time)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr

    # Stage 4: L-BFGS covering radius
    points, cr = _lbfgs_refine(points, max_time=t4, start_time=start_time, beta=50.0)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr

    # Stage 5: Lloyd again
    points, cr = _spherical_lloyd(points, max_iters=60, time_limit=t5, start_time=start_time)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr

    # Stage 6: L-BFGS covering radius with higher beta
    points, cr = _lbfgs_refine(points, max_time=t6, start_time=start_time, beta=100.0)

    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 1:
            return points, cr

    # Stage 7: Spherical GD final polish
    points, cr = _spherical_gd(points, max_iters=500, lr_init=0.004,
                               time_limit=t7, start_time=start_time)
    return points, cr

def _repulsion_refine(points, max_time=15.0, start_time=None, s=8.0, maxiter=200):
    """
    Refine by minimizing repulsive Riesz energy to spread points more evenly.
    Uses a moderate exponent to balance spreading vs. local optimization.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norms[x_norms < 1e-15] = 1.0
        x = x / x_norms
        val, grad = _riesz_energy_grad(x, s=s)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 400}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _multi_beta_lse_refine(points, max_time=25.0, start_time=None):
    """
    Cascade of L-BFGS refinements with increasing beta values for the LSE surrogate.
    Lower beta gives smoother landscape for coarse optimization; higher beta
    sharpens toward the true max-facet circumradius.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    betas = [30.0, 60.0, 120.0, 250.0]
    for beta in betas:
        if time.time() - start_time > max_time - 3:
            break
        remaining_beta = max_time - (time.time() - start_time)
        if remaining_beta < 3:
            break
        pts, cr = _lbfgs_refine(points, max_time=min(remaining_beta, 12.0),
                                start_time=start_time, beta=beta)
        if cr < best_cr:
            best = pts.copy()
            best_cr = cr
        points = best.copy()

    return best, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci(N, seed=1, jitter=0.03),
        _jittered_fibonacci(N, seed=7, jitter=0.05),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
        _random_points(N, seed=200),
        _jittered_fibonacci(N, seed=42, jitter=0.02),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_per = min(remaining / (len(initial_configs) - i), 15.0)
        refined, cr = _refine(init_pts, time_limit=time_per, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.15, 12.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    remaining = deadline - time.time()
    if remaining > 10:
        fin_time = min(remaining * 0.30, 22.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    remaining = deadline - time.time()
    if remaining > 8:
        riesz_time = max(remaining * 0.20, 4.0)
        riesz_pts, riesz_cr = _lbfgs_riesz(best_points, max_time=riesz_time,
                                           start_time=start_time, s=16.0, maxiter=400)
        if riesz_cr < best_cr:
            best_cr = riesz_cr
            best_points = riesz_pts.copy()

    remaining = deadline - time.time()
    if remaining > 6:
        rep_time = max(remaining * 0.15, 4.0)
        rep_pts, rep_cr = _repulsion_refine(best_points, max_time=rep_time,
                                            start_time=start_time, s=8.0)
        if rep_cr < best_cr:
            best_cr = rep_cr
            best_points = rep_pts.copy()

    remaining = deadline - time.time()
    if remaining > 5:
        mbl_time = max(remaining * 0.35, 8.0)
        mbl_pts, mbl_cr = _multi_beta_lse_refine(best_points, max_time=mbl_time,
                                                 start_time=start_time)
        if mbl_cr < best_cr:
            best_cr = mbl_cr
            best_points = mbl_pts.copy()

    remaining = deadline - time.time()
    if remaining > 4:
        gram_time = max(remaining * 0.15, 3.0)
        gram_pts, gram_cr = _lbfgs_gram(best_points, max_time=gram_time,
                                        start_time=start_time, beta=30.0)
        if gram_cr < best_cr:
            best_cr = gram_cr
            best_points = gram_pts.copy()

    remaining = deadline - time.time()
    if remaining > 3:
        gd_time = max(remaining * 0.5, 1.5)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
