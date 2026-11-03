"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        n_samples = 20000
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

def _grad_covering_radius(points):
    """
    Compute a gradient of the covering radius w.r.t. the points (in R^3,
    then projected to tangent space). Uses the top-K largest facets with
    a soft weighting to get a smooth, sub-gradient-like direction.
    """
    facets = _fast_facets(points)
    if not facets:
        return np.zeros_like(points), 0.0
    facets.sort(key=lambda x: -x[0])
    max_r = facets[0][0]
    K = min(8, len(facets))
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
    """Spherical gradient descent with adaptive step size."""
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
    """
    Lloyd-style: each point moves to the average circumcenter of its adjacent facets.
    """
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
    """Simulated annealing with random perturbations of a few points."""
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

def _lbfgs_smooth_minimize(points, time_limit=None, start_time=None, beta=50.0, maxiter=300):
    """
    Use L-BFGS-B to minimize a log-sum-exp smooth surrogate of the covering radius.
    This is the key improvement from the inspiration: a smooth differentiable
    objective that guides gradient-based optimization more effectively than
    the nonsmooth max-of-facets radius.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best_points = points.copy()
    best_cr = _covering_radius(points)

    def _facet_data(pts):
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None
        centers = []
        radii = []
        facets = []
        for simplex in hull.simplices:
            if len(simplex) != 3:
                continue
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
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
            centers.append(v)
            radii.append(r)
            facets.append(simplex)
        if not centers:
            return None, None, None
        return np.array(centers), np.array(radii), facets

    def _objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        x = x / norms

        centers, radii, facets = _facet_data(x)
        if centers is None:
            return 100.0, np.zeros_like(x_flat)

        scaled = radii * beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max = log_sum_exp / beta

        weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))

        grad_flat = np.zeros_like(x_flat)
        for i in range(len(centers)):
            center = centers[i]
            weight = weights[i]
            simplex = facets[i]
            for idx in simplex:
                p = x[idx]
                cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
                sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
                if sin_r < 1e-10:
                    continue
                grad_p = -center / sin_r
                grad_p_tan = grad_p - np.dot(grad_p, p) * p
                grad_flat[idx*3:(idx+1)*3] += weight * grad_p_tan

        return soft_max, grad_flat

    def _time_check():
        if time_limit is not None and start_time is not None:
            return (time.time() - start_time) < time_limit
        return True

    # Wrap the objective with time checking
    obj_count = [0]
    def _obj_with_time(x_flat):
        obj_count[0] += 1
        if not _time_check():
            raise ValueError("Time limit exceeded")
        val, grad = _objective_and_grad(x_flat)
        return val, grad

    try:
        result = minimize(
            _obj_with_time,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8, 'maxfun': 1000}
        )
        if result.success or result.fun < best_cr:
            new_pts = result.x.reshape(N, 3)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_cr = _covering_radius(new_pts)
            if new_cr < best_cr:
                best_points = new_pts
                best_cr = new_cr
    except Exception:
        pass

    return best_points, best_cr

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd -> GD -> L-BFGS smooth -> Lloyd -> GD."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    # Stage 1: Lloyd
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, _covering_radius(points)
        t1 = min(rem * 0.25, 6)
    else:
        t1 = 6
    points, cr = _spherical_lloyd(points, max_iters=120, time_limit=t1, start_time=start_time)

    # Stage 2: Gradient descent
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t2 = min(rem * 0.3, 10)
    else:
        t2 = 10
    points, cr = _spherical_gd(points, max_iters=1500, lr_init=0.008,
                               time_limit=t2, start_time=start_time)

    # Stage 3: L-BFGS smooth minimization (key improvement)
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t3 = min(rem * 0.25, 12)
    else:
        t3 = 12
    points, cr = _lbfgs_smooth_minimize(points, time_limit=t3, start_time=start_time, beta=50.0, maxiter=200)

    # Stage 4: Lloyd again
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t4 = min(rem * 0.1, 4)
    else:
        t4 = 4
    points, cr = _spherical_lloyd(points, max_iters=80, time_limit=t4, start_time=start_time)

    # Stage 5: Final GD polish
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 1:
            return points, cr
        t5 = max(rem * 0.5, 1.0)
    else:
        t5 = 5
    points, cr = _spherical_gd(points, max_iters=1000, lr_init=0.004,
                               time_limit=t5, start_time=start_time)
    return points, cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Refine each initial config with a time budget
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 8:
            break
        time_per = min(remaining / (len(initial_configs) - i), 18.0)
        refined, cr = _refine(init_pts, time_limit=time_per, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 2: Simulated annealing from best
    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.3, 15.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 3: Final refinement with L-BFGS
    remaining = deadline - time.time()
    if remaining > 8:
        fin_time = min(remaining * 0.5, 15.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Phase 4: Final L-BFGS polish with higher beta for tighter convergence
    remaining = deadline - time.time()
    if remaining > 5:
        lbfgs_time = max(remaining * 0.4, 3.0)
        lb_pts, lb_cr = _lbfgs_smooth_minimize(best_points, time_limit=lbfgs_time,
                                               start_time=start_time, beta=100.0, maxiter=300)
        if lb_cr < best_cr:
            best_cr = lb_cr
            best_points = lb_pts.copy()

    # Final GD polish with whatever time is left
    remaining = deadline - time.time()
    if remaining > 3:
        gd_time = max(remaining * 0.7, 2.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=2000, lr_init=0.003,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
