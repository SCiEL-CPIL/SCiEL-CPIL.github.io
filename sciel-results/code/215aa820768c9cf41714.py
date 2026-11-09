import numpy as np
from scipy.optimize import minimize, minimize as _minimize_nm
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

    # === Strategy 1: Edge-length parametrization via circumcircle ===
    # For a convex polygon inscribed in a circle of radius R, the central angle
    # subtended by edge i is alpha_i = 2*arcsin(l_i/(2R)).
    # Closure requires sum(alpha_i) = 2*pi, which determines R uniquely.
    # We parametrize by 7 free edge lengths (the 8th is determined by perimeter),
    # solve for R via bisection, and construct the polygon.
    # This is a 7-dimensional smooth optimization in the strictly positive interior.

    def solve_radius(edge_lengths):
        """Find R such that sum(2*arcsin(l_i/(2R))) = 2*pi."""
        # For R -> infinity, arcsin(l_i/(2R)) ~ l_i/(2R), so sum ~ sum(l_i)/(2R) -> 0
        # For R -> max(l_i)/2 from above, one term -> pi, others are small
        # We need sum = 2*pi, so R must be large enough.
        # Bisection: f(R) = sum(2*arcsin(l_i/(2R))) - 2*pi, f is decreasing in R.
        l_max = np.max(edge_lengths)
        R_lo = l_max / 2.0 * 1.001  # just above the minimum
        R_hi = 10.0 * np.sum(edge_lengths)  # large enough upper bound
        
        # Check if solution exists
        def f(R):
            if R <= l_max / 2.0:
                return 2 * np.pi  # arcsin would be > 1, so this is a penalty
            vals = edge_lengths / (2.0 * R)
            if np.any(vals > 1.0):
                return 2 * np.pi
            return np.sum(2.0 * np.arcsin(vals)) - 2.0 * np.pi
        
        if f(R_lo) < 0:
            # No solution with this configuration (edges too long relative to each other)
            return None
        
        # Bisection
        for _ in range(100):
            R_mid = 0.5 * (R_lo + R_hi)
            fm = f(R_mid)
            if abs(fm) < 1e-14:
                break
            if fm > 0:
                R_lo = R_mid
            else:
                R_hi = R_mid
        
        R = 0.5 * (R_lo + R_hi)
        if f(R) > 1e-8:
            return None
        return R

    def build_from_edges(edge_lengths):
        """Construct convex polygon from 8 edge lengths using circumcircle."""
        R = solve_radius(edge_lengths)
        if R is None:
            return None
        
        # Central angles
        alphas = 2.0 * np.arcsin(edge_lengths / (2.0 * R))
        
        # Verify sum is approximately 2*pi
        if abs(np.sum(alphas) - 2.0 * np.pi) > 1e-8:
            return None
        
        # Construct vertices on circle of radius R, starting at angle 0
        angles = np.zeros(8)
        angles[1:] = np.cumsum(alphas[:-1])
        
        pts = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
        
        # Center at origin (translate)
        pts = pts - np.mean(pts, axis=0)
        
        return pts

    def obj_edges(x):
        """Objective: negative log product of distances, given 7 free edge lengths.
        The 8th edge length is determined by fixed perimeter."""
        # x = [l_0, ..., l_6], l_7 = perimeter - sum(l_0..l_6)
        # Use log parametrization for positivity: x_i = log(l_i)
        l_free = np.exp(x)
        # Fixed total perimeter P (choose P=1 for now, scale doesn't matter)
        P = 1.0
        l_7 = P - np.sum(l_free)
        if l_7 <= 1e-6:
            return 1e10
        
        edges = np.concatenate([l_free, [l_7]])
        pts = build_from_edges(edges)
        if pts is None:
            return 1e10
        
        return -log_sum_dist(pts)

    # Initialize: regular octagon has all edges equal
    # For regular octagon with perimeter P, each edge = P/8
    P_init = 1.0
    l_eq = P_init / 8.0
    edges_init_log = np.full(7, np.log(l_eq))

    # Bounds: each edge length in [1e-4, P-7e-4] to ensure positivity
    log_lo = np.log(1e-4)
    log_hi = np.log(P_init - 7 * 1e-4)
    edges_bounds = [(log_lo, log_hi)] * 7

    if time.time() - start_time < deadline:
        # Try multiple initializations
        inits = [
            edges_init_log.copy(),
            edges_init_log + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1]) * 0.5,
            edges_init_log + np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]) * 0.5,
        ]
        for init in inits:
            if time.time() - start_time > deadline:
                break
            try:
                res = minimize(obj_edges, init, method='SLSQP',
                               bounds=edges_bounds,
                               options={'maxiter': 1000, 'ftol': 1e-14})
                if res.fun < 1e9:
                    l_free = np.exp(res.x)
                    l_7 = P_init - np.sum(l_free)
                    if l_7 > 1e-6:
                        edges = np.concatenate([l_free, [l_7]])
                        pts = build_from_edges(edges)
                        if pts is not None:
                            s = compute_score(pts)
                            if s > best_score:
                                best_score = s
                                best_points = pts
            except Exception:
                pass

    # === Strategy 1b: Nelder-Mead polish on the exact objective ===
    # The exact score is non-smooth at the diameter constraint (max of distances).
    # NM doesn't need gradients and can escape flat regions where SLSQP stalls.
    def nm_obj(x):
        pts = x.reshape(8, 2)
        return -compute_score(pts)

    def nm_polish(pts0, maxiter=3000):
        try:
            x0 = pts0.flatten()
            res = _minimize_nm(nm_obj, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-12,
                                        'fatol': 1e-12, 'adaptive': True})
            pts = res.x.reshape(8, 2)
            # validate distinctness
            d2 = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            m = np.triu(np.ones((n, n), dtype=bool), k=1)
            if np.min(d2[m]) < 1e-8:
                return None
            return pts
        except Exception:
            return None

    if time.time() - start_time < deadline:
        # Polish the current best with NM
        polished = nm_polish(best_points, maxiter=4000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 1c: High-precision soft-diameter polish ===
    # Use a log-sum-exp soft-max surrogate for the diameter to get a smooth objective
    # that is close to the true diameter when beta is large.
    def soft_diameter(pts, beta=100.0):
        """Soft max of distances using log-sum-exp."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        d = dists[mask]
        d_max = np.max(d)
        # Log-sum-exp: log(sum(exp(beta * d))) / beta
        # = d_max + log(sum(exp(beta * (d - d_max)))) / beta
        shifted = beta * (d - d_max)
        # Avoid overflow
        shifted = np.clip(shifted, -500, 500)
        lse = d_max + np.log(np.sum(np.exp(shifted))) / beta
        return lse

    def soft_score(pts, beta=100.0):
        """Score using soft diameter."""
        D_soft = soft_diameter(pts, beta)
        if D_soft < 1e-15:
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        scaled = dists * (2.0 / D_soft)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        log_prod = np.sum(np.log(np.maximum(scaled[mask], 1e-300)))
        return np.exp(log_prod) / (n ** n)

    def soft_obj(x, beta=100.0):
        pts = x.reshape(8, 2)
        return -soft_score(pts, beta)

    if time.time() - start_time < deadline:
        # Progressive beta refinement
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() - start_time > deadline:
                break
            try:
                x0 = best_points.flatten()
                res = minimize(soft_obj, x0, args=(beta,), method='BFGS',
                               options={'maxiter': 500, 'gtol': 1e-12})
                pts = res.x.reshape(8, 2)
                # Validate
                d2 = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
                m = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.min(d2[m]) < 1e-8:
                    continue
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass

    # === Strategy 2: Symmetric with x-axis mirror ===
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

    # === Strategy 2b: NM polish on symmetric solutions found so far ===
    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=4000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 3: Full 16-DOF optimization ===
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

    # === Strategy 4: Random perturbations + local optimize + NM polish ===
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

    # === Strategy 4b: NM polish the best symmetric result ===
    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=5000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 5: Perturbed octagon full optimization ===
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

    # === Strategy 5b: NM polish the best full result ===
    if time.time() - start_time < deadline:
        polished = nm_polish(best_points, maxiter=5000)
        if polished is not None:
            s = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished

    # === Strategy 6: Tighter SLSQP polish on the best (exact objective via log) ===
    if time.time() - start_time < deadline:
        try:
            x0 = best_points.flatten()
            res = minimize(obj_full, x0, method='SLSQP',
                           bounds=full_bounds, constraints=full_constraints,
                           options={'maxiter': 1000, 'ftol': 1e-14})
            pts = res.x.reshape(8, 2)
            s = compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
            # then NM polish once more
            polished = nm_polish(pts, maxiter=6000)
            if polished is not None:
                s2 = compute_score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished
        except Exception:
            pass

    # === Final high-precision polish ===
    if time.time() - start_time < deadline:
        # Very tight SLSQP on the exact objective
        def obj_final(x):
            pts = x.reshape(8, 2)
            return -log_sum_dist(pts)

        final_constraints = []
        for i in range(8):
            for j in range(i + 1, 8):
                def c(x, i=i, j=j):
                    pts = x.reshape(8, 2)
                    return 2.0 - np.linalg.norm(pts[i] - pts[j])
                final_constraints.append({'type': 'ineq', 'fun': c})

        final_bounds = [(-1.5, 1.5), (-1.5, 1.5)] * 8

        try:
            x0 = best_points.flatten()
            res = minimize(obj_final, x0, method='SLSQP',
                           bounds=final_bounds, constraints=final_constraints,
                           options={'maxiter': 2000, 'ftol': 1e-16})
            if res.fun < 1e9:
                pts = res.x.reshape(8, 2)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass

        # Final NM polish with very tight tolerances
        if time.time() - start_time < deadline:
            polished = nm_polish(best_points, maxiter=8000)
            if polished is not None:
                s = compute_score(polished)
                if s > best_score:
                    best_score = s
                    best_points = polished

        # One more soft-diameter pass with very high beta
        if time.time() - start_time < deadline:
            try:
                x0 = best_points.flatten()
                res = minimize(soft_obj, x0, args=(500.0,), method='BFGS',
                               options={'maxiter': 1000, 'gtol': 1e-14})
                pts = res.x.reshape(8, 2)
                d2 = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
                m = np.triu(np.ones((n, n), dtype=bool), k=1)
                if np.min(d2[m]) < 1e-8:
                    pts = best_points
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
