import numpy as np
from scipy.optimize import minimize
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

    # === Strategy 1: Edge-length parametrization (NEW) ===
    # Parametrize convex polygon by 8 edge lengths.
    # Given edge lengths l_0..l_7, construct convex polygon via:
    #   - Compute turning angles from edge lengths (for a convex polygon inscribed
    #     in a circle, the central angle subtended by edge i is 2*arcsin(l_i/(2R)))
    #   - But we need to find R such that the polygon closes.
    # Alternative: use the fact that for a convex polygon with given side lengths,
    # there's a unique (up to similarity) convex polygon. We can construct it by
    # placing vertices sequentially with appropriate turning angles.
    # 
    # Simpler approach: parametrize by 7 angles (turning angles) that sum to 2*pi,
    # and optimize both angles and a scale, with edge lengths derived from the
    # requirement that the polygon closes. Actually, let's use a different approach:
    #
    # For a convex n-gon, we can parametrize by the n-1 independent turning angles
    # (the nth is determined by sum = 2*pi) and the n edge lengths. But that's 2n-1
    # params. Instead, let's parametrize by the 8 edge lengths and solve for the
    # unique convex polygon (up to rotation/translation) with those side lengths.
    #
    # Actually, the cleanest approach: parametrize by 7 positive numbers (log edge
    # lengths) and construct the convex polygon via the "turning angle" method.
    # For a convex polygon with side lengths s_0, ..., s_{n-1}, the turning angles
    # theta_i satisfy: the polygon closes iff sum of vectors s_i * exp(i*cumsum(theta)) = 0.
    # This is a system of 2 equations in n-1 unknowns (since sum theta_i = 2*pi).
    #
    # Simpler: use the fact that any convex polygon can be represented as the intersection
    # of half-planes, but that's complex. Let me use a direct construction:
    #
    # Given 8 edge lengths (l_0, ..., l_7) > 0, construct a convex octagon by:
    # 1. Start at origin, first edge along x-axis.
    # 2. At each vertex, turn by angle theta_i where sum(theta_i) = 2*pi.
    # 3. The theta_i are determined by the requirement that the polygon closes.
    #    This gives 2 equations (x and y closure) in 7 unknowns (theta_0..theta_6,
    #    with theta_7 = 2*pi - sum(theta_0..theta_6)).
    #
    # This is underdetermined. Instead, let's parametrize by the 8 turning angles
    # (sum = 2*pi) and optimize them directly, with edge lengths as free parameters.
    # Total: 7 + 8 = 15 params, but we can fix scale (diameter = 2), so 14 params.
    #
    # Even simpler: just optimize the 8 turning angles (7 free, sum = 2*pi) and
    # 8 edge lengths (7 free, scale fixed by diameter=2). That's 14 DOF.
    #
    # Actually, let me try a different, cleaner approach:
    # Parametrize by 7 positive numbers a_0..a_6. Let edge lengths be:
    #   l_i = a_i / sum(a) * L for i=0..6, l_7 = L * (1 - sum(a_0..a_6)/sum(a))
    # Hmm, this is getting complicated. Let me just use the full 16-DOF with
    # better initialization and constraints, plus the symmetric approach.
    #
    # Let me try: parametrize by 8 angles phi_0..phi_7 (turning angles, sum=2*pi)
    # and 8 edge lengths l_0..l_7. Construct the polygon, compute score.
    # Use 7 free angles (phi_7 = 2*pi - sum(phi_0..phi_6)) and 8 free lengths.
    # Then normalize by diameter. Total: 15 params.

    def build_from_edges_angles(x):
        """x = [phi_0..phi_6, l_0..l_7], construct convex polygon."""
        phis = x[:7]
        phis_full = np.concatenate([phis, [2*np.pi - np.sum(phis)]])
        lengths = x[7:]
        
        # Ensure all phis are positive (for convexity)
        if np.any(phis_full <= 1e-6) or np.any(lengths <= 1e-6):
            return None
        
        # Construct polygon
        pts = np.zeros((8, 2))
        angle = 0.0
        for i in range(8):
            if i > 0:
                angle += phis_full[i-1]
            if i < 7:
                pts[i+1] = pts[i] + lengths[i] * np.array([np.cos(angle), np.sin(angle)])
            else:
                # Last edge should close the polygon (approximately)
                pass
        
        # Actually, the last edge is implicit. Let me redo:
        # Start at origin. Edge i goes from vertex i to vertex i+1.
        # Direction of edge i is sum of turning angles before it.
        pts = np.zeros((8, 2))
        cum_angle = 0.0
        for i in range(7):
            pts[i+1] = pts[i] + lengths[i] * np.array([np.cos(cum_angle), np.sin(cum_angle)])
            cum_angle += phis_full[i]
        
        # The 8th edge goes from pts[7] back to pts[0] = origin
        # Its length should be ||pts[7]|| and its direction should be consistent
        # For a proper convex polygon, pts[7] to origin should have the right length
        # and the turning angle at vertex 7 should be phis_full[7]
        
        # Check closure: the last edge length is ||pts[7] - pts[0]||
        last_len = np.linalg.norm(pts[7] - pts[0])
        if last_len < 1e-6:
            return None
        
        # The polygon is pts[0..7] with the implicit closing edge
        return pts

    def obj_edges_angles(x):
        pts = build_from_edges_angles(x)
        if pts is None:
            return 1e10
        # Check that the implicit last edge length matches
        last_len = np.linalg.norm(pts[7] - pts[0])
        if last_len < 1e-6 or abs(last_len - x[14]) > 0.5 * x[14]:
            return 1e10
        return -log_sum_dist(pts)

    # Initialize with regular octagon edge lengths and equal turning angles
    reg = regular_octagon()
    reg_edges = np.array([np.linalg.norm(reg[i+1] - reg[i]) for i in range(7)])
    reg_edges = np.concatenate([reg_edges, [np.linalg.norm(reg[0] - reg[7])]])
    reg_phis = np.full(7, 2*np.pi/8)
    edges_init = np.concatenate([reg_phis, reg_edges])

    # Add constraint: all turning angles positive, all edge lengths positive
    # And the implicit last edge length should match
    def closure_constraint(x):
        pts = build_from_edges_angles(x)
        if pts is None:
            return -1e10
        last_len = np.linalg.norm(pts[7] - pts[0])
        return last_len - x[14]

    def pos_angle_i(i):
        def c(x):
            return x[i] - 1e-4
        return c

    def pos_edge_i(i):
        def c(x):
            return x[7+i] - 1e-4
        return c

    edges_constraints = []
    for i in range(7):
        edges_constraints.append({'type': 'ineq', 'fun': pos_angle_i(i)})
    for i in range(8):
        edges_constraints.append({'type': 'ineq', 'fun': pos_edge_i(i)})

    edges_bounds = [(1e-4, np.pi - 1e-4)] * 7 + [(1e-4, 3.0)] * 8

    if time.time() - start_time < deadline:
        for init_scale in [1.0, 0.8, 1.2]:
            if time.time() - start_time > deadline:
                break
            init = edges_init * init_scale
            try:
                res = minimize(obj_edges_angles, init, method='SLSQP',
                               bounds=edges_bounds, constraints=edges_constraints,
                               options={'maxiter': 500, 'ftol': 1e-10})
                if res.success:
                    pts = build_from_edges_angles(res.x)
                    if pts is not None:
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

    # === Strategy 4: Random perturbations + local optimize ===
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
