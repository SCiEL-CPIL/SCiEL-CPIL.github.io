import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])

    def calc_diam(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    def log_score(pts):
        """Sum of log distances over all unordered pairs."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        d_ij = np.maximum(d_ij, 1e-15)
        return np.sum(np.log(d_ij))

    def compute_score(pts):
        """Delta / n^n after scaling to diameter 2."""
        pts = np.array(pts, dtype=float)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        if np.any(d < 1e-9):
            return -np.inf
        diam = np.max(d)
        if diam < 1e-9:
            return -np.inf
        pts_scaled = pts * (2.0 / diam)
        diff_s = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists_s = np.sqrt(np.sum(diff_s**2, axis=2))
        d_s = dists_s[mask]
        log_delta = np.sum(np.log(d_s))
        delta = np.exp(log_delta)
        return delta / (n ** n)

    # --- Angular representation: 8 points on unit circle with variable gaps ---
    # gaps[i] is the angular gap from point i to point i+1 (mod 8)
    # Sum of gaps = 2*pi, all gaps > 0
    # We optimize 7 free gaps (the 8th is determined)

    def gaps_to_points(gaps7):
        """Convert 7 free gaps to 8 points on unit circle."""
        gaps = np.append(gaps7, 2*np.pi - np.sum(gaps7))
        # Ensure all gaps positive; if not, clip
        gaps = np.maximum(gaps, 1e-6)
        # Normalize to sum to 2*pi
        gaps = gaps / np.sum(gaps) * 2*np.pi
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = angles[i-1] + gaps[i-1]
        # Rotate so that the diameter (max chord) is horizontal
        # Find the pair with max distance
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Center at origin (already is)
        # Rotate to make the diameter pair horizontal
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        if i_max != j_max:
            # Rotate so that the line from pts[i_max] to pts[j_max] is horizontal
            mid_angle = 0.5*(angles[i_max] + angles[j_max])
            # We want the diameter to be along x-axis
            # The midpoint of the arc should be at angle 0 or pi
            # Let's rotate by -mid_angle
            rot = -mid_angle
            angles = angles + rot
            pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts, gaps

    def angular_objective(gaps7):
        """
        Minimize -log_score of the point set, with penalty if diameter > 2.
        Actually, since we're on unit circle, diameter is at most 2 (diameter of circle).
        The max chord length on unit circle is 2 (when points are antipodal).
        So diameter <= 2 always holds for points on unit circle.
        We just need to maximize log_score.
        """
        pts, gaps = gaps_to_points(gaps7)
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d = dists[mask]
        if np.any(d < 1e-8):
            return 1e10
        return -log_score(pts)

    def angular_gradient(gaps7):
        """Numerical gradient of angular_objective."""
        eps = 1e-8
        grad = np.zeros(7)
        f0 = angular_objective(gaps7)
        for i in range(7):
            gp = gaps7.copy()
            gp[i] += eps
            fp = angular_objective(gp)
            grad[i] = (fp - f0) / eps
        return grad

    best_score = -np.inf
    best_pts = None

    def update_best(pts):
        nonlocal best_score, best_pts
        # Scale to diameter 2
        d = calc_diam(pts)
        if d < 1e-10:
            return
        pts_scaled = pts * (2.0 / d)
        # Check distinctness
        diffs = pts_scaled[:, np.newaxis, :] - pts_scaled[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        mask = ~np.eye(n, dtype=bool)
        d_s = dists[mask]
        if np.any(d_s < 1e-8):
            return
        score = compute_score(pts_scaled)
        if score > best_score:
            best_score = score
            best_pts = pts_scaled.copy()

    # --- 1. Regular octagon ---
    theta = 2 * np.pi * np.arange(n) / n
    octagon = np.column_stack([np.cos(theta), np.sin(theta)])
    update_best(octagon)

    # --- 2. Optimize angular gaps ---
    # Start from regular octagon gaps: all = 2*pi/8 = pi/4
    regular_gaps = np.full(7, 2*np.pi/n)

    # Multiple random restarts with different initial gap patterns
    rng = np.random.RandomState(42)
    restarts = [
        regular_gaps.copy(),
        # Slightly perturbed regular
        regular_gaps + rng.normal(0, 0.05, 7),
        # Two-cluster pattern: 4 points close, 4 points close
        np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
        # Alternating large/small gaps
        np.array([0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.8]),
        # Random
        rng.uniform(0.2, 1.0, 7),
        # Another random
        rng.uniform(0.2, 1.0, 7),
    ]

    for init_gaps in restarts:
        if time.time() > deadline - 20:
            break
        # Ensure gaps are positive and sum to less than 2*pi
        init_gaps = np.maximum(init_gaps, 0.05)
        s = np.sum(init_gaps)
        if s >= 2*np.pi:
            init_gaps = init_gaps * (2*np.pi - 0.1) / s

        try:
            res = minimize(
                angular_objective,
                init_gaps,
                jac=angular_gradient,
                method='L-BFGS-B',
                bounds=[(0.05, 2*np.pi - 0.1)] * 7,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < 1e9:
                pts, _ = gaps_to_points(res.x)
                update_best(pts)
        except Exception:
            pass

    # --- 3. Full 16D optimization from best angular result ---
    # Use the best points found as starting point for full coordinate optimization
    if best_pts is not None:
        if time.time() < deadline - 15:
            # Optimize in full 16D space with diameter penalty
            def full_objective(x):
                pts = x.reshape(n, 2)
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                d_ij = dists[iu]
                if np.any(d_ij < 1e-8):
                    return 1e10
                val_log = np.sum(np.log(d_ij))
                # Penalty for diameter > 2
                diam = np.max(dists)
                pen = 0.0
                if diam > 2.0:
                    pen = (diam - 2.0)**2 * 1000
                return -val_log + pen

            def full_gradient(x):
                pts = x.reshape(n, 2)
                diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs**2, axis=2))
                d_ij = dists[iu]
                if np.any(d_ij < 1e-8):
                    return np.zeros(16)

                grad = np.zeros_like(pts)
                for idx in range(num_pairs):
                    i, j = iu[0][idx], iu[1][idx]
                    d = d_ij[idx]
                    diff = pts[i] - pts[j]
                    g = diff / (d * d)
                    grad[i] += g
                    grad[j] -= g

                # Gradient of penalty
                diam = np.max(dists)
                if diam > 2.0:
                    # Find which pair achieves max
                    i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
                    if i_max != j_max:
                        diff = pts[i_max] - pts[j_max]
                        d_max = dists[i_max, j_max]
                        g_pen = 2.0 * (d_max - 2.0) * 1000 * (diff / d_max)
                        grad[i_max] += g_pen
                        grad[j_max] -= g_pen

                return -grad.flatten()

            try:
                res = minimize(
                    full_objective,
                    best_pts.flatten(),
                    jac=full_gradient,
                    method='L-BFGS-B',
                    bounds=[(-3, 3)] * 16,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < 1e9:
                    pts = res.x.reshape(n, 2)
                    update_best(pts)
            except Exception:
                pass

    # --- 4. Final polish with multiple L-BFGS-B restarts ---
    if best_pts is not None:
        if time.time() < deadline - 5:
            # Try a few more perturbations
            rng2 = np.random.RandomState(123)
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
                perturbed = best_pts + rng2.normal(0, 0.01, (n, 2))
                try:
                    res = minimize(
                        full_objective,
                        perturbed.flatten(),
                        jac=full_gradient,
                        method='L-BFGS-B',
                        bounds=[(-3, 3)] * 16,
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.fun < 1e9:
                        pts = res.x.reshape(n, 2)
                        update_best(pts)
                except Exception:
                    pass

    # Ensure valid output
    if best_pts is None:
        best_pts = octagon

    # Scale to diameter 2
    d = calc_diam(best_pts)
    if d > 1e-10:
        best_pts = best_pts * (2.0 / d)

    # Check distinctness and fix if needed
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                best_pts[j] += np.array([1e-6, 1e-6])

    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e5:
        best_pts = best_pts * (1e5 / max_coord)
        d = calc_diam(best_pts)
        if d > 1e-10:
            best_pts = best_pts * (2.0 / d)

    return [[float(p[0]), float(p[1])] for p in best_pts]
