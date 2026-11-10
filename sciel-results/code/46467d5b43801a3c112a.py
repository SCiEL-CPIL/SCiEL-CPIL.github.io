import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    def true_score(points):
        """Compute (d_min/d_max)^2 using squared distances."""
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def neg_score(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return -true_score(pts)

    # Smooth surrogate: maximize min distance by minimizing sum of inverse powers
    # Uses analytic gradient via chain rule through RMS normalization
    def smooth_obj(x_flat, power=4.0):
        pts = x_flat.reshape(n, d)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms < 1e-6:
            return 1e10
        pts_n = pts / rms
        dists_sq = pdist(pts_n, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        return np.sum(dists_sq ** (-power/2.0))

    def smooth_obj_grad(x_flat, power=4.0):
        pts = x_flat.reshape(n, d)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms < 1e-6:
            return np.zeros_like(x_flat)
        pts_n = pts / rms
        dists_sq = pdist(pts_n, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        
        # Gradient of sum(d_i^(-power/2)) w.r.t. dists_sq
        # d/d(dist_sq) [dist_sq^(-power/2)] = (-power/2) * dist_sq^(-power/2 - 1)
        d_dsq = (-power/2.0) * dists_sq ** (-power/2.0 - 1.0)
        
        # Now we need gradient w.r.t. pts_n, then chain through normalization
        # pdist gives distances in order: (0,1),(0,2),...,(0,n-1),(1,2),...
        # We need to map each pair distance back to its contributing points
        grad_pts_n = np.zeros((n, d))
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = pts_n[i] - pts_n[j]
                # d(dist_sq)/d(pts_n[i]) = 2*diff, d(dist_sq)/d(pts_n[j]) = -2*diff
                grad_pts_n[i] += d_dsq[idx] * 2.0 * diff
                grad_pts_n[j] += d_dsq[idx] * (-2.0) * diff
                idx += 1
        
        # Chain rule through RMS normalization: pts_n = pts / rms
        # d(pts_n)/d(pts) = (1/rms) * I - (pts * pts^T) / (rms^3 * n)
        # More precisely: d(pts_n[k])/d(pts[m]) = delta_km/rms - pts_n[k]*pts_n[m]/(rms*n)
        # Gradient w.r.t. pts: grad_pts = (grad_pts_n / rms) - (pts_n @ grad_pts_n) * pts / (rms * n)
        # Wait, let me be more careful.
        # pts_n = pts / rms, where rms = sqrt(mean(sum(pts^2, axis=1)))
        # d(pts_n)/d(pts) is a 3n x 3n matrix.
        # grad_pts = J^T @ grad_pts_n where J = d(pts_n)/d(pts)
        # J[i,j] for i-th point's k-th coord vs j-th point's l-th coord
        # pts_n[i,k] = pts[i,k] / rms
        # d(pts_n[i,k])/d(pts[j,l]) = delta_ik * delta_jl / rms - pts[i,k] * pts[j,l] / (rms^3 * n)
        # So grad_pts[i,k] = grad_pts_n[i,k] / rms - (sum_{j,l} grad_pts_n[j,l] * pts[j,l]) * pts[i,k] / (rms^3 * n)
        # Let s = sum of element-wise product of grad_pts_n and pts
        s = np.sum(grad_pts_n * pts)
        grad_pts = grad_pts_n / rms - s * pts / (rms**3 * n)
        
        return grad_pts.flatten()

    # Log-sum-exp based soft-max for the max distance (epigraph-like)
    def lse_obj(x_flat, beta=50.0):
        """Minimize soft-max of distances (proxy for d_max) while keeping min distance high.
        Actually, we want to maximize d_min/d_max. Let's use:
        f = -log(sum(exp(beta * d_i))) + beta * d_min_proxy
        Simpler: minimize (soft_max - soft_min) to make all distances equal.
        """
        pts = x_flat.reshape(n, d)
        rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if rms < 1e-6:
            return 1e10
        pts_n = pts / rms
        dists_sq = pdist(pts_n, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        dists = np.sqrt(dists_sq)
        
        # Soft max and soft min via log-sum-exp
        max_d = np.max(dists)
        min_d = np.min(dists)
        # Objective: minimize (soft_max - soft_min) to equalize distances
        # soft_max = (1/beta) * log(sum(exp(beta * d_i)))
        # soft_min = -(1/beta) * log(sum(exp(-beta * d_i)))
        # We want to minimize soft_max - soft_min, which encourages all distances to be similar
        shifted_max = dists - max_d
        soft_max = max_d + (1.0/beta) * np.log(np.sum(np.exp(beta * shifted_max)))
        shifted_min = -dists + max_d  # = max_d - dists
        soft_min = -(max_d) + (1.0/beta) * np.log(np.sum(np.exp(-beta * (dists - max_d))))
        # Actually: soft_min = min_d approx = -(1/beta)*log(sum(exp(-beta*d_i)))
        # = -(1/beta)*log(sum(exp(beta*(max_d - d_i)))) = max_d - (1/beta)*log(sum(exp(beta*(max_d-d_i))))
        # Let's just compute directly:
        soft_max = (1.0/beta) * np.log(np.sum(np.exp(beta * (dists - max_d)))) + max_d
        soft_min = -(1.0/beta) * np.log(np.sum(np.exp(-beta * (dists - min_d)))) + min_d
        
        return soft_max - soft_min

    # Best tracking
    best_score = 0.0
    best_points = None

    def try_update(points):
        nonlocal best_score, best_points
        if points is None:
            return
        points = normalize_rms(points)
        if not np.all(np.isfinite(points)):
            return
        s = true_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # =================================================================
    # PHASE 1: Generate diverse initial candidates
    # =================================================================
    candidates = []

    # --- Model A: Two antipodal rings ---
    def make_anticipodal_rings(h, r1, r2, theta1, twist):
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta1
        ring1 = np.column_stack([
            r1 * np.cos(angles1),
            r1 * np.sin(angles1),
            np.full(7, h)
        ])
        angles2 = angles1 + twist
        ring2 = np.column_stack([
            r2 * np.cos(angles2),
            r2 * np.sin(angles2),
            np.full(7, -h)
        ])
        pts = np.vstack([ring1, ring2])
        return normalize_rms(pts)

    h_vals = np.linspace(0.0, 1.0, 15)
    r_ratio_vals = np.linspace(0.2, 1.0, 12)
    theta1_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)
    twist_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)

    for h in h_vals:
        if time.time() > deadline - 50:
            break
        for rr in r_ratio_vals:
            for th1 in theta1_vals:
                for tw in twist_vals:
                    pts = make_anticipodal_rings(h, 1.0, rr, th1, tw)
                    candidates.append(pts)

    # --- Model B: icosahedron + 2 poles ---
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(normalize_rms(np.vstack([ico, poles])))

    # --- Model C: Cube + octahedron ---
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    candidates.append(normalize_rms(np.vstack([cube, octa])))

    # --- Model D: Random antipodal ---
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        pts = normalize_rms(np.vstack([pts_upper, pts_lower]))
        candidates.append(pts)

    # --- Model E: Fibonacci sphere points ---
    def fibonacci_sphere(n_pts):
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_pts):
            z = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(1.0 - z**2)
            theta = golden_angle * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points, dtype=float)

    fib_pts = fibonacci_sphere(14)
    fib_pts = fib_pts / np.linalg.norm(fib_pts, axis=1, keepdims=True)
    candidates.append(normalize_rms(fib_pts))

    for angle in [np.pi/6, np.pi/4, np.pi/3]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = fib_pts @ rot.T
        candidates.append(normalize_rms(rotated))

    # --- Model F: Iterative repulsion ---
    for seed in range(10):
        rng = np.random.RandomState(1000 + seed)
        pts = rng.randn(n, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        for it in range(200):
            dists = pdist(pts, metric='sqeuclidean')
            forces = np.zeros((n, 3))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-10:
                        continue
                    force = diff / (dist_sq * np.sqrt(dist_sq))
                    forces[i] += force
            pts = pts + 0.01 * forces
            pts = normalize_rms(pts)
        candidates.append(pts)

    # Score all candidates
    for cand in candidates:
        try_update(cand)

    # =================================================================
    # PHASE 2: Local optimization from best candidates
    # =================================================================
    if best_points is not None:
        polish_list = [best_points]

        cand_scores = [(true_score(c), c) for c in candidates]
        cand_scores.sort(key=lambda x: -x[0])
        for s, c in cand_scores[1:8]:
            if s > best_score * 0.75:
                polish_list.append(c)

        for init_pts in polish_list:
            if time.time() > deadline - 60:
                break

            init_pts = normalize_rms(init_pts)
            x0 = init_pts.flatten()

            # Step 2a: L-BFGS-B on smooth surrogate
            try:
                res1 = minimize(
                    lambda x: smooth_obj(x, 4.0),
                    x0,
                    jac=lambda x: smooth_obj_grad(x, 4.0),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts1 = normalize_rms(res1.x.reshape(n, d))
            except Exception:
                pts1 = init_pts

            # Step 2b: Nelder-Mead on true objective
            try:
                res2 = minimize(
                    neg_score,
                    pts1.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                             'adaptive': True, 'maxfev': 10000}
                )
                pts2 = normalize_rms(res2.x.reshape(n, d))
                try_update(pts2)
            except Exception:
                pts2 = pts1

            # Step 2c: More NM iterations if time allows
            if time.time() < deadline - 70:
                try:
                    res3 = minimize(
                        neg_score,
                        pts2.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                 'adaptive': True, 'maxfev': 20000}
                    )
                    pts3 = normalize_rms(res3.x.reshape(n, d))
                    try_update(pts3)
                except Exception:
                    pass

    # =================================================================
    # PHASE 3: Differential evolution for global search
    # =================================================================
    if time.time() < deadline - 100:
        try:
            bounds = [(-2, 2)] * (n * d)
            res_de = differential_evolution(
                neg_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=30,
                tol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='sobol',
                update_interval=1
            )
            pts_de = normalize_rms(res_de.x.reshape(n, d))
            try_update(pts_de)
        except Exception:
            pass

    # =================================================================
    # PHASE 4: Fine-tuning with random restarts + local polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 40:
        rng = np.random.RandomState(123)
        for restart in range(30):
            if time.time() > deadline - 20:
                break
            noise_scale = 0.1 * (1.0 - restart / 40.0)
            perturbed = best_points + rng.randn(n, d) * noise_scale
            perturbed = normalize_rms(perturbed)

            try:
                res = minimize(
                    neg_score,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True, 'maxfev': 5000}
                )
                pts_r = normalize_rms(res.x.reshape(n, d))
                try_update(pts_r)
            except Exception:
                continue

    # =================================================================
    # PHASE 5: Final ultra-fine polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True, 'maxfev': 50000}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            try_update(pts_final)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    if not np.all(np.isfinite(best_points)):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    return best_points
