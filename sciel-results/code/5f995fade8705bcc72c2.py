import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 8

    def fast_log_score(x):
        """x: 16-vector (8 points, 2 coords each). Returns log(score) or -inf."""
        pts = x.reshape(8, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(8, k=1)
        d = dists[iu]
        if np.any(d < 1e-12):
            return -np.inf
        D = np.max(d)
        if D < 1e-12:
            return -np.inf
        sum_log_d = np.sum(np.log(d))
        log_score = 2.0 * (sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)) - 8.0 * np.log(8.0)
        return log_score

    def objective(x):
        """For minimization: return -log_score."""
        if time.time() > deadline:
            return 1e10
        s = fast_log_score(x)
        if np.isneginf(s) or np.isnan(s):
            return 1e10
        return -s

    def refine(x0, max_iter=300):
        """Run Powell then Nelder-Mead from x0, return best (score, points)."""
        if time.time() > deadline - 5.0:
            return -np.inf, x0.reshape(8, 2)
        
        best_s = -np.inf
        best_x = x0.copy()
        
        # Try Powell
        try:
            res_p = minimize(
                objective, x0, method='Powell',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            s_p = -res_p.fun
            if np.isfinite(s_p) and s_p > -1e9:
                best_s = s_p
                best_x = res_p.x
        except Exception:
            pass

        # Try Nelder-Mead on the result of Powell
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(
                    objective, best_x, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                s_nm = -res_nm.fun
                if np.isfinite(s_nm) and s_nm > best_s:
                    best_s = s_nm
                    best_x = res_nm.x
            except Exception:
                pass

        return best_s, best_x.reshape(8, 2)

    best_score = -np.inf
    best_points = None

    # Strategy 1: Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    s_oct = fast_log_score(pts_oct.flatten())
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct.copy()

    # Strategy 2: Parametric two-square model
    # Points: 4 from square 1 (radius R1, angle 0), 4 from square 2 (radius R2, angle theta)
    def two_squares_param(params):
        R1, R2, theta = params
        a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1*np.cos(a1), R1*np.sin(a1)])
        a2 = a1 + theta
        p2 = np.column_stack([R2*np.cos(a2), R2*np.sin(a2)])
        return np.vstack([p1, p2]).flatten()

    def param_obj(params):
        if time.time() > deadline - 10.0:
            return 1e10
        x = two_squares_param(params)
        s = fast_log_score(x)
        if np.isneginf(s) or np.isnan(s):
            return 1e10
        return -s

    # Run DE on the 3-parameter manifold
    if time.time() < deadline - 30.0:
        try:
            bounds = [(0.1, 3.0), (0.1, 3.0), (0.0, np.pi/2)]
            de_res = differential_evolution(
                param_obj, bounds,
                maxiter=200, popsize=30, tol=1e-8,
                seed=42, init='sobol', workers=1
            )
            if de_res.fun < 1e9:
                best_params = de_res.x
                x_de = two_squares_param(best_params)
                s_de = -de_res.fun
                if s_de > best_score:
                    best_score = s_de
                    best_points = x_de.reshape(8, 2)
                
                # Refine the DE result with full 16-param optimization
                # This allows breaking the square symmetry if beneficial
                s_ref, pts_ref = refine(x_de, max_iter=500)
                if s_ref > best_score:
                    best_score = s_ref
                    best_points = pts_ref.copy()
        except Exception:
            pass

    # Strategy 3: Grid search over two-square parameters (coarse) then refine
    # This catches cases where DE might get stuck or if the landscape is tricky
    if time.time() < deadline - 20.0:
        R1_vals = np.linspace(0.5, 2.5, 8)
        R2_vals = np.linspace(0.5, 2.5, 8)
        theta_vals = np.linspace(0.0, np.pi/4, 8)
        
        candidates = []
        for R1 in R1_vals:
            for R2 in R2_vals:
                for th in theta_vals:
                    x = two_squares_param([R1, R2, th])
                    s = fast_log_score(x)
                    if np.isfinite(s):
                        candidates.append((s, x))
        
        candidates.sort(key=lambda t: -t[0])
        top_k = min(10, len(candidates))
        for s_init, x_init in candidates[:top_k]:
            if time.time() > deadline - 15.0:
                break
            s_ref, pts_ref = refine(x_init)
            if s_ref > best_score:
                best_score = s_ref
                best_points = pts_ref.copy()

    # Strategy 4: Random multi-start full 16-param optimization
    rng = np.random.default_rng(123)
    for trial in range(20):
        if time.time() > deadline - 10.0:
            break
        # Generate diverse random configurations
        if trial % 3 == 0:
            # Near-regular octagon with perturbation
            a = np.linspace(0, 2*np.pi, 8, endpoint=False) + rng.uniform(-0.1, 0.1, 8)
            r = 1.0 + rng.uniform(-0.2, 0.2, 8)
            pts = np.column_stack([r*np.cos(a), r*np.sin(a)])
        elif trial % 3 == 1:
            # Two squares with random params
            R1 = rng.uniform(0.5, 2.5)
            R2 = rng.uniform(0.5, 2.5)
            th = rng.uniform(0, np.pi/4)
            pts = two_squares_param([R1, R2, th]).reshape(8, 2)
        else:
            # Fully random
            pts = rng.uniform(-2, 2, (8, 2))

        s_init = fast_log_score(pts.flatten())
        if np.isneginf(s_init):
            continue
        s_ref, pts_ref = refine(pts.flatten(), max_iter=200)
        if s_ref > best_score:
            best_score = s_ref
            best_points = pts_ref.copy()

    # Final polish
    if best_points is not None and time.time() < deadline - 5.0:
        s_final, pts_final = refine(best_points.flatten(), max_iter=500)
        if s_final > best_score:
            best_score = s_final
            best_points = pts_final

    if best_points is None:
        best_points = pts_oct

    # Scale to diameter 2
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    D = np.max(dists)
    if D > 0:
        best_points = best_points * (2.0 / D)

    return [[float(p[0]), float(p[1])] for p in best_points]
