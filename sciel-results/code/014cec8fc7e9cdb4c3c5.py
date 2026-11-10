import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 620  # safety margin before 630s kill

    n = 14
    d = 3

    def score(pts):
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return (dd.min() / dd.max()) ** 2

    def neg_metric(x):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        return -score(pts)

    def make_nm(x0, maxiter=20000, xatol=1e-12, fatol=1e-14):
        r = minimize(neg_metric, x0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': xatol,
                              'fatol': fatol, 'adaptive': True})
        return r.x

    def make_lbfgs(x0, maxiter=1000):
        r = minimize(neg_metric, x0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
        return r.x

    # Differential Evolution on a smooth surrogate
    def log_sum_exp_surrogate(x):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        dd = np.maximum(dd, 1e-10)
        # Minimize sum of 1/dd^2 (Coulomb-like) to spread points
        return np.sum(1.0 / dd**2)

    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(42)

    def update_best(pts):
        nonlocal best_pts, best_score
        pts = pts - np.mean(pts, axis=0)
        s = score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # ---- Strategy 1: Icosahedral symmetry parameterization ----
    phi = (1 + np.sqrt(5)) / 2
    I12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    I12 = I12 / np.linalg.norm(I12, axis=1, keepdims=True)
    R0 = 1.0 / np.sqrt(1 + phi**2)

    def make_points_ico(p):
        r = p[0]
        pts = np.vstack([r * I12, [[0, 0, p[1]], [0, 0, p[2]]]])
        return pts

    def neg_metric_ico(p):
        pts = make_points_ico(p)
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return -(dd.min() / dd.max()) ** 2

    def make_nm_ico(p0, maxiter=30000):
        r = minimize(neg_metric_ico, p0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-14,
                              'fatol': 1e-16, 'adaptive': True})
        return r.x

    def make_lbfgs_ico(p0, maxiter=4000):
        r = minimize(neg_metric_ico, p0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-14})
        return r.x

    ico_candidates = []
    # Symmetric z: z1 = -z2 = a
    for a in np.linspace(0.0, 0.6, 13):
        ico_candidates.append(np.array([R0, -a, a]))
    # Slightly asymmetric
    for a in np.linspace(0.1, 0.5, 8):
        for da in [0.0, 0.01, -0.01, 0.02, -0.02]:
            ico_candidates.append(np.array([R0, -a + da, a]))
    # Vary outer radius
    for r in [R0*0.95, R0*0.97, R0, R0*1.01, R0*1.03]:
        for a in np.linspace(0.0, 0.5, 7):
            ico_candidates.append(np.array([r, -a, a]))
    # Random
    rng_ico = np.random.RandomState(7)
    for _ in range(40):
        r = R0 * (0.9 + 0.2 * rng_ico.rand())
        a = rng_ico.rand() * 0.6
        da = (rng_ico.rand() - 0.5) * 0.05
        ico_candidates.append(np.array([r, -a + da, a]))

    for p in ico_candidates:
        if time.time() > DEADLINE - 80:
            break
        try:
            p1 = make_lbfgs_ico(p)
        except Exception:
            p1 = p
        try:
            p2 = make_nm_ico(p1)
        except Exception:
            p2 = p1
        for pp in [p1, p2]:
            pts = make_points_ico(pp)
            update_best(pts)

    # ---- Strategy 2: Differential Evolution on smooth surrogate ----
    def de_surrogate(x):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        dd = np.maximum(dd, 1e-10)
        # Use log-sum-exp for smoothness: minimize -log(sum(exp(-dd^2/k)))
        # which approximates maximizing the minimum distance
        k = 0.01
        vals = -dd**2 / k
        vals = vals - vals.max()  # numerical stability
        return -np.log(np.sum(np.exp(vals)))

    def de_neg_metric(x):
        # For DE: maximize (dmin/dmax)^2, so minimize negative
        pts = x.reshape((n, d))
        dd = pdist(pts)
        if dd.max() <= 0:
            return 1.0
        return -(dd.min() / dd.max()) ** 2

    # Use scipy's differential_evolution
    def make_de(bounds, maxiter=100, popsize=15, tol=1e-10):
        from scipy.optimize import differential_evolution
        r = differential_evolution(de_surrogate, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    tol=tol, seed=42, polish=False)
        return r.x

    # Run DE with several random restarts on the surrogate
    num_de_starts = 10
    for i in range(num_de_starts):
        if time.time() > DEADLINE - 100:
            break
        # Random initial population via DE
        x0 = rng.randn(n * d) * 0.5
        # Use L-BFGS on surrogate first to get a good starting point
        try:
            r_coulomb = minimize(de_surrogate, x0, method='L-BFGS-B',
                                 options={'maxiter': 500, 'ftol': 1e-10})
            x_coulomb = r_coulomb.x
        except Exception:
            x_coulomb = x0

        # Polish with NM on the true metric
        try:
            x_nm = make_nm(x_coulomb, maxiter=15000)
        except Exception:
            x_nm = x_coulomb

        pts = x_nm.reshape((n, d))
        update_best(pts)

    # ---- Strategy 3: Direct DE on the true metric (smaller pop) ----
    if time.time() < DEADLINE - 60:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-3, 3)] * (n * d)
            r_de = differential_evolution(de_neg_metric, bounds,
                                           maxiter=50, popsize=12,
                                           tol=1e-8, seed=99, polish=False)
            pts = r_de.x.reshape((n, d))
            update_best(pts)
            # Polish
            if time.time() < DEADLINE - 30:
                try:
                    x_final = make_nm(r_de.x, maxiter=20000)
                    pts_final = x_final.reshape((n, d))
                    update_best(pts_final)
                except Exception:
                    pass
        except Exception:
            pass

    # ---- Strategy 4: Structural seeds ----
    # Cube + Octahedron
    c = 1.0
    o = 2.0
    cube = np.array([[c, c, c], [c, c, -c], [c, -c, c], [c, -c, -c],
                     [-c, c, c], [-c, c, -c], [-c, -c, c], [-c, -c, -c]])
    octa = np.array([[o, 0, 0], [-o, 0, 0], [0, o, 0], [0, -o, 0],
                     [0, 0, o], [0, 0, -o]])
    x_seed = np.vstack([cube, octa]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed, maxiter=15000)
            pts = x_nm.reshape((n, d))
            update_best(pts)
        except Exception:
            pass

    # Distorted icosahedron + poles
    poles = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    x_seed2 = np.vstack([I12, poles]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed2, maxiter=15000)
            pts = x_nm.reshape((n, d))
            update_best(pts)
        except Exception:
            pass

    # ---- Strategy 5: Final polish with longer NM on best result ----
    if best_pts is not None and time.time() < DEADLINE - 10:
        x_best = best_pts.flatten()
        try:
            x_final = make_nm(x_best, maxiter=50000, xatol=1e-13, fatol=1e-15)
            pts_final = x_final.reshape((n, d))
            update_best(pts_final)
        except Exception:
            pass

    if best_pts is None:
        best_pts = np.zeros((n, d))
        for i in range(n):
            best_pts[i] = rng.randn(d)
        best_pts = best_pts - np.mean(best_pts, axis=0)

    if not np.all(np.isfinite(best_pts)):
        best_pts = np.vstack([I12, [[0, 0, -0.25], [0, 0, 0.25]]])
        best_pts = best_pts - np.mean(best_pts, axis=0)

    return best_pts
