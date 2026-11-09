import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def regular_octagon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_score(pts):
        """Compute Delta/n^n after scaling to diameter 2."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D < 1e-15:
            return 0.0
        scaled = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        log_prod = np.sum(np.log(np.maximum(scaled[mask], 1e-300)))
        return np.exp(log_prod) / (n ** n)

    def log_sum_dist(pts):
        """Sum of log distances over unique pairs (for optimization)."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        if np.any(d < 1e-10):
            return -1e10
        return np.sum(np.log(d))

    def max_dist(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return np.max(dists[mask])

    # Best tracking
    best_points = regular_octagon()
    best_score = compute_score(best_points)

    # === Strategy 1: Symmetric with x-axis mirror ===
    def build_sym(x):
        pts = np.zeros((8, 2))
        pts[0] = [-1.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [x[0], x[1]]
        pts[3] = [x[0], -x[1]]
        pts[4] = [x[2], x[3]]
        pts[5] = [x[2], -x[3]]
        pts[6] = [x[4], x[5]]
        pts[7] = [x[4], -x[5]]
        return pts

    def obj_sym(x):
        pts = build_sym(x)
        return -log_sum_dist(pts)

    def sym_constraint(i, j):
        def c(x):
            pts = build_sym(x)
            return 2.0 - np.linalg.norm(pts[i] - pts[j])
        return c

    sym_constraints = []
    for i in [2, 4, 6]:
        sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(0, i)})
        sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(1, i)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 4)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 6)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(4, 6)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 5)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(2, 7)})
    sym_constraints.append({'type': 'ineq', 'fun': sym_constraint(4, 7)})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[1]})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[3]})
    sym_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - x[5]})

    sym_bounds = [(-1.5, 1.5), (1e-4, 1.0),
                  (-1.5, 1.5), (1e-4, 1.0),
                  (-1.5, 1.5), (1e-4, 1.0)]

    sym_inits = [
        np.array([0.707, 0.707, 0.0, 1.0, -0.707, 0.707]),
        np.array([0.5, 0.8, -0.3, 0.9, -0.8, 0.5]),
        np.array([0.0, 1.0, 0.8, 0.6, -0.8, 0.6]),
        np.array([0.9, 0.3, 0.0, 1.0, -0.9, 0.3]),
        np.array([0.3, 0.95, -0.5, 0.85, -0.95, 0.4]),
    ]

    for init in sym_inits:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(obj_sym, init, method='SLSQP',
                           bounds=sym_bounds, constraints=sym_constraints,
                           options={'maxiter': 300, 'ftol': 1e-12})
            if res.success:
                pts = build_sym(res.x)
                d = max_dist(pts)
                if 1e-5 < np.min(np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)[np.triu(np.ones((n,n), dtype=bool), k=1)]):
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
        except Exception:
            pass

    # === Strategy 2: Full 16-DOF optimization ===
    def obj_full(x):
        pts = x.reshape(8, 2)
        return -log_sum_dist(pts)

    full_constraints = []
    for i in range(8):
        for j in range(i+1, 8):
            def c(x, i=i, j=j):
                pts = x.reshape(8, 2)
                return 2.0 - np.linalg.norm(pts[i] - pts[j])
            full_constraints.append({'type': 'ineq', 'fun': c})

    full_bounds = [(-1.5, 1.5), (-1.5, 1.5)] * 8

    oct_pts = regular_octagon()
    full_init = oct_pts.flatten()

    if time.time() - start_time < deadline:
        try:
            res = minimize(obj_full, full_init, method='SLSQP',
                           bounds=full_bounds, constraints=full_constraints,
                           options={'maxiter': 400, 'ftol': 1e-11})
            if res.success:
                pts = res.x.reshape(8, 2)
                d = max_dist(pts)
                if d > 1e-5:
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
        except Exception:
            pass

    # === Strategy 3: Random perturbations + local optimize ===
    np.random.seed(123)
    if time.time() - start_time < deadline:
        for trial in range(15):
            if time.time() - start_time > deadline:
                break
            x0 = np.random.uniform(-0.9, 0.9, 3)
            y0 = np.random.uniform(0.15, 0.95, 3)
            init_rand = np.concatenate([x0, y0])

            try:
                res = minimize(obj_sym, init_rand, method='SLSQP',
                               bounds=sym_bounds, constraints=sym_constraints,
                               options={'maxiter': 200, 'ftol': 1e-11})
                if res.success:
                    pts = build_sym(res.x)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass

    # === Strategy 4: Perturbed octagon full optimization ===
    if time.time() - start_time < deadline:
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for _ in range(5):
            if time.time() - start_time > deadline:
                break
            pert = angles + np.random.uniform(-0.3, 0.3, 8)
            pts_init = np.column_stack([np.cos(pert), np.sin(pert)])
            x_init = pts_init.flatten()

            try:
                res = minimize(obj_full, x_init, method='SLSQP',
                               bounds=full_bounds, constraints=full_constraints,
                               options={'maxiter': 300, 'ftol': 1e-11})
                if res.success:
                    pts = res.x.reshape(8, 2)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
            except Exception:
                pass

    # === Strategy 5: Differential evolution for global search ===
    if time.time() - start_time < deadline:
        def de_obj(x):
            pts = x.reshape(8, 2)
            # Penalize if any distance > 2 (diameter constraint)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if np.max(dists[mask]) > 2.0 + 1e-6:
                return -log_sum_dist(pts) + 100 * (np.max(dists[mask]) - 2.0)
            if np.min(dists[mask]) < 1e-10:
                return -1e10
            return -log_sum_dist(pts)

        de_bounds = [(-1.5, 1.5), (-1.5, 1.5)] * 8
        try:
            de_res = differential_evolution(de_obj, de_bounds, seed=42,
                                            maxiter=100, popsize=15,
                                            tol=1e-8, mutation=(0.3, 1.0),
                                            recombination=0.8, polish=True)
            if de_res.success:
                pts = de_res.x.reshape(8, 2)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

    # === Strategy 6: Nelder-Mead polish on best solution ===
    if time.time() - start_time < deadline:
        def nm_obj(x):
            pts = x.reshape(8, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            if np.max(dists[mask]) > 2.0 + 1e-6:
                return -log_sum_dist(pts) + 100 * (np.max(dists[mask]) - 2.0)
            if np.min(dists[mask]) < 1e-10:
                return 1e10
            return -log_sum_dist(pts)

        try:
            nm_res = minimize(nm_obj, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if nm_res.success:
                pts = nm_res.x.reshape(8, 2)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

    # Final validation
    diffs = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.min(dists[mask]) < 1e-6:
        best_points = regular_octagon()

    if np.any(np.abs(best_points) > 1e6):
        D = max_dist(best_points)
        if D > 0:
            best_points = best_points * (1e5 / D)

    return best_points.tolist()
