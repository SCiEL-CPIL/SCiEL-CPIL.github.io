import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    start = time.time()
    DEADLINE = start + 620
    n, d = 14, 3

    def score(pts):
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        return (dd.min() / dd.max()) ** 2

    def neg_metric(x):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        return -score(pts)

    def make_nm(x0, maxiter=20000, xatol=1e-13, fatol=1e-15):
        r = minimize(neg_metric, x0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': xatol,
                              'fatol': fatol, 'adaptive': True})
        return r.x

    def make_lbfgs(x0, maxiter=1000):
        r = minimize(neg_metric, x0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15})
        return r.x

    best_pts = None
    best_score = -1.0
    rng = np.random.RandomState(42)

    # Smooth surrogate
    def log_sum_exp(x, beta=100):
        x_max = np.max(x)
        if x_max * beta > 700:
            return x_max + np.log(len(x))
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

    def soft_min_max(x, beta=100):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        if dd.max() <= 0:
            return 0.0
        soft_dmin = -log_sum_exp(-dd, beta) / beta
        soft_dmax = log_sum_exp(dd, beta) / beta
        if soft_dmax <= 0:
            return 0.0
        return -(soft_dmin / soft_dmax) ** 2

    def neg_metric_soft(x, beta=100):
        return soft_min_max(x, beta)

    def make_nm_soft(x0, beta=100, maxiter=5000):
        def obj(x):
            return soft_min_max(x, beta)
        r = minimize(obj, x0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-13,
                              'fatol': 1e-15, 'adaptive': True})
        return r.x

    # Icosahedral symmetry
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

    def make_nm_ico(p0, maxiter=20000):
        r = minimize(neg_metric_ico, p0, method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
        return r.x

    def make_lbfgs_ico(p0, maxiter=4000):
        r = minimize(neg_metric_ico, p0, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-15})
        return r.x

    ico_candidates = []
    for a in np.linspace(0.0, 0.6, 13):
        ico_candidates.append(np.array([R0, -a, a]))
    for a in np.linspace(0.1, 0.5, 8):
        for da in [0.0, 0.01, -0.01, 0.02, -0.02]:
            ico_candidates.append(np.array([R0, -a + da, a]))
    for r in [R0*0.95, R0*0.97, R0, R0*1.01, R0*1.03]:
        for a in np.linspace(0.0, 0.5, 7):
            ico_candidates.append(np.array([r, -a, a]))
    rng_ico = np.random.RandomState(7)
    for _ in range(40):
        r = R0 * (0.9 + 0.2 * rng_ico.rand())
        a = rng_ico.rand() * 0.6
        da = (rng_ico.rand() - 0.5) * 0.05
        ico_candidates.append(np.array([r, -a + da, a]))

    for p in ico_candidates:
        if time.time() > DEADLINE - 60:
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
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    # Coulomb repulsion + NM
    def coulomb_obj(x):
        pts = x.reshape((n, d))
        dd = pdist(pts)
        dd = np.where(dd < 1e-10, 1e-10, dd)
        return np.sum(1.0 / dd)

    def make_coulomb(x0):
        r = minimize(coulomb_obj, x0, method='L-BFGS-B',
                     options={'maxiter': 1000, 'ftol': 1e-12})
        return r.x

    num_starts = 25
    for i in range(num_starts):
        if time.time() > DEADLINE - 120:
            break
        x0 = rng.randn(n * d)
        try:
            x_coulomb = make_coulomb(x0)
        except Exception:
            x_coulomb = x0
        try:
            x_nm = make_nm(x_coulomb)
        except Exception:
            x_nm = x_coulomb
        pts = x_nm.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        s = score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Structural seeds
    c = 1.0
    o = 2.0
    cube = np.array([[c, c, c], [c, c, -c], [c, -c, c], [c, -c, -c],
                     [-c, c, c], [-c, c, -c], [-c, -c, c], [-c, -c, -c]])
    octa = np.array([[o, 0, 0], [-o, 0, 0], [0, o, 0], [0, -o, 0], [0, 0, o], [0, 0, -o]])
    x_seed = np.vstack([cube, octa]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    poles = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    x_seed2 = np.vstack([I12, poles]).flatten()
    if time.time() < DEADLINE - 30:
        try:
            x_nm = make_nm(x_seed2)
            pts = x_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

    # Differential Evolution on smooth surrogate
    if time.time() < DEADLINE - 120:
        try:
            bounds = [(-3, 3)] * (n * d)
            de_result = differential_evolution(
                lambda x: soft_min_max(x, beta=100),
                bounds=bounds,
                seed=42,
                maxiter=400,
                popsize=30,
                tol=1e-10,
                polish=False,
                init='latinhypercube'
            )
            x_de = de_result.x
            x_de_nm = make_nm(x_de, maxiter=30000)
            pts = x_de_nm.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        except Exception:
            pass

        if time.time() < DEADLINE - 60 and best_pts is not None:
            try:
                de_result2 = differential_evolution(
                    lambda x: soft_min_max(x, beta=20),
                    bounds=bounds,
                    seed=123,
                    maxiter=300,
                    popsize=30,
                    tol=1e-10,
                    polish=False,
                    init='latinhypercube'
                )
                x_de2 = de_result2.x
                x_de2_nm = make_nm(x_de2, maxiter=30000)
                pts2 = x_de2_nm.reshape((n, d))
                pts2 = pts2 - np.mean(pts2, axis=0)
                s2 = score(pts2)
                if s2 > best_score:
                    best_score = s2
                    best_pts = pts2.copy()
            except Exception:
                pass

    # Multi-stage beta annealing polish
    if best_pts is not None and time.time() < DEADLINE - 60:
        x_best = best_pts.flatten()
        for beta in [20, 50, 100, 200, 500]:
            if time.time() > DEADLINE - 30:
                break
            try:
                x_annealed = make_nm_soft(x_best, beta=beta, maxiter=5000)
                x_annealed = make_nm(x_annealed, maxiter=10000)
                pts = x_annealed.reshape((n, d))
                pts = pts - np.mean(pts, axis=0)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
                    x_best = pts.flatten()
            except Exception:
                pass

    # Final high-precision polish
    if best_pts is not None and time.time() < DEADLINE - 5:
        x_best = best_pts.flatten()
        try:
            x_final1 = make_nm(x_best, maxiter=50000, xatol=1e-14, fatol=1e-16)
            pts_final1 = x_final1.reshape((n, d))
            pts_final1 = pts_final1 - np.mean(pts_final1, axis=0)
            s_final1 = score(pts_final1)

            if time.time() < DEADLINE - 2:
                x_final2 = make_nm(pts_final1.flatten(), maxiter=50000, xatol=1e-15, fatol=1e-17)
                pts_final2 = x_final2.reshape((n, d))
                pts_final2 = pts_final2 - np.mean(pts_final2, axis=0)
                s_final2 = score(pts_final2)

                if s_final2 > s_final1:
                    s_final = s_final2
                    pts_final = pts_final2
                else:
                    s_final = s_final1
                    pts_final = pts_final1
            else:
                s_final = s_final1
                pts_final = pts_final1

            if s_final > best_score:
                best_score = s_final
                best_pts = pts_final.copy()
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
