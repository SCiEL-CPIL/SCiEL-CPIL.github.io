"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # --- 1. Generate a good initial configuration (hexagonal lattice) ---
    def _hex_init(n):
        # Hexagonal packing with spacing tuned to fit ~n circles in unit disk
        # Area of unit disk = pi. Area per circle ~ pi * r^2.
        # For hex packing, density ~ pi/(2*sqrt(3)) ≈ 0.9069
        # So n * pi * r^2 ≈ 0.9069 * pi => r^2 ≈ 0.9069/n => r ≈ 0.952/sqrt(n)
        # For n=300: r ≈ 0.952/17.32 ≈ 0.055
        # Spacing d = 2r ≈ 0.11
        r_est = 0.952 / np.sqrt(n)
        d = 2.0 * r_est
        # Generate hex lattice points
        pts = []
        half = 1.0 / d + 2
        for row in range(-int(half), int(half) + 1):
            y = row * d * np.sqrt(3) / 2.0
            x_offset = (d / 2.0) * (row % 2)
            for col in range(-int(half), int(half) + 1):
                x = col * d + x_offset
                if x * x + y * y <= (1.0 - r_est) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) < n:
            # Fallback: random points in disk
            pts = _random_init(n)
        else:
            # Select n points closest to center for better symmetry
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            pts = pts[idx]
        return pts

    def _random_init(n):
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def _compute_radius(pts):
        if len(pts) < 2:
            return 0.0
        tree = cKDTree(pts)
        # Find closest pair
        pairs = tree.query_pairs(2.0)  # 2.0 is generous upper bound
        if len(pairs) == 0:
            min_dist = np.inf
        else:
            min_dist = np.inf
            for i, j in pairs:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        min_dist = min(min_dist, np.inf)
        # Container constraint
        dists_to_center = np.hypot(pts[:, 0], pts[:, 1])
        min_container = 1.0 - np.max(dists_to_center)
        r = min(min_dist / 2.0, min_container)
        return max(0.0, r)

    def _residuals(x, n):
        pts = x.reshape(-1, 2)
        # Soft-min of pair distances and container distances
        # We want to maximize r = min(min_pair_dist/2, min_container_dist)
        # Equivalent to minimizing -r, but let's use a smooth surrogate
        # Residuals: for each pair, d_ij - 2*t (want d_ij >= 2*t)
        # For each point, 1 - |p_i| - t (want 1 - |p_i| >= t)
        # But least_squares minimizes sum of squares of residuals
        # Let's use: residuals = [d_ij/2 - t for all pairs, (1-|p_i|) - t for all i]
        # But t is not in x. Alternative: just use the distances and let the optimizer find good spacing.
        # Better approach: use the actual radius as a measure.
        # Let's use a different formulation: minimize the negative of a soft-min.
        
        # Simple approach: residuals are the "violations" of a target radius
        # But we don't know the target. Let's just use the pair distances and container distances.
        # Minimize the variance of distances? No.
        
        # Let's use: for each pair (i,j), residual = |p_i - p_j|
        # For each point i, residual = 1 - |p_i|
        # And we want all these to be large. So minimize negative?
        # least_squares minimizes sum(residuals^2). If we use residuals = -distance, 
        # it will try to make distances small, which is wrong.
        
        # Better: use a fixed target r and penalize violations.
        # Or: use the log-sum-exp trick to create a smooth approximation of -min.
        
        # Let's try: residual for pair = max(0, 2*r_target - d_ij)
        # residual for container = max(0, r_target - (1-|p_i|))
        # But r_target changes.
        
        # Alternative: just minimize the sum of squares of (1/d_ij) and (1/(1-|p_i|))
        # This pushes all distances to be large.
        
        pts = x.reshape(-1, 2)
        n_pts = len(pts)
        res = []
        # Pair distances
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                d = np.linalg.norm(pts[i] - pts[j])
                d = max(d, 1e-10)
                res.append(1.0 / d)
        # Container distances
        for i in range(n_pts):
            dc = 1.0 - np.linalg.norm(pts[i])
            dc = max(dc, 1e-10)
            res.append(1.0 / dc)
        return np.array(res)

    # --- 2. Initial configuration ---
    np.random.seed(42)
    pts = _hex_init(N)
    x0 = pts.flatten()
    r0 = _compute_radius(pts)
    
    best_pts = pts.copy()
    best_r = r0

    # --- 3. Optimization using least_squares ---
    # The residual function is expensive (O(N^2)). For N=300, that's ~45k pairs.
    # Let's use a more efficient residual: only use nearest neighbors.
    
    def _residuals_efficient(x, n):
        pts = x.reshape(-1, 2)
        tree = cKDTree(pts)
        # Query k nearest neighbors for each point
        k = min(10, n - 1)
        dists, idxs = tree.query(pts, k=k + 1)  # +1 includes self
        dists = dists[:, 1:]  # Remove self
        idxs = idxs[:, 1:]
        
        res = []
        # Use only the closest few neighbors to reduce cost
        for i in range(n):
            for j in range(min(5, len(idxs[i]))):
                d = dists[i, j]
                d = max(d, 1e-10)
                res.append(1.0 / d)
        
        # Container distances
        dc = 1.0 - np.linalg.norm(pts, axis=1)
        dc = np.maximum(dc, 1e-10)
        res.extend(1.0 / dc)
        
        return np.array(res)

    # Run optimization with time budget
    max_iters = 200
    try:
        result = least_squares(
            _residuals_efficient,
            x0,
            args=(N,),
            method='trf',
            max_nfev=max_iters,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10
        )
        if result.success or result.fun < np.inf:
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_pts

    # --- 4. Local refinement with gradient-based approach ---
    # Use a simple iterative relaxation: move each point away from its nearest neighbor
    # and toward the center if too close to boundary.
    
    def _relax(pts, n, n_iters=1000, step_size=0.01):
        pts = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline:
                break
            tree = cKDTree(pts)
            for i in range(n):
                # Find nearest neighbor
                d, j = tree.query(pts[i])
                if j == i:
                    # Find second nearest
                    d, idxs = tree.query(pts[i], k=2)
                    j = idxs[1]
                    d = d[1]
                
                if d < 1e-10:
                    continue
                
                # Move away from nearest neighbor
                dir_vec = (pts[i] - pts[j]) / d
                pts[i] += step_size * dir_vec
                
                # Move toward center if too close to boundary
                r_i = np.linalg.norm(pts[i])
                if r_i > 0.95:
                    dir_to_center = -pts[i] / r_i
                    pts[i] += step_size * 0.5 * dir_to_center
                
                # Keep inside unit disk
                r_i = np.linalg.norm(pts[i])
                if r_i > 1.0:
                    pts[i] = pts[i] / r_i * 0.999
        return pts

    pts_relaxed = _relax(best_pts, N, n_iters=500, step_size=0.005)
    r_relaxed = _compute_radius(pts_relaxed)
    if r_relaxed > best_r:
        best_pts = pts_relaxed
        best_r = r_relaxed

    if time.time() > deadline:
        return best_pts

    # --- 5. Second round of least_squares from improved initial guess ---
    try:
        x0 = best_pts.flatten()
        result = least_squares(
            _residuals_efficient,
            x0,
            args=(N,),
            method='trf',
            max_nfev=100,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12
        )
        if result.fun < np.inf:
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass

    # --- 6. Final polish: small step relaxation ---
    if time.time() < deadline - 5.0:
        pts_polished = _relax(best_pts, N, n_iters=200, step_size=0.002)
        r_polished = _compute_radius(pts_polished)
        if r_polished > best_r:
            best_pts = pts_polished
            best_r = r_polished

    # Ensure valid output
    best_pts = np.asarray(best_pts, dtype=np.float64)
    # Verify all points are inside unit disk
    norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = norms < 1.0
    if not np.all(mask):
        best_pts = best_pts[mask]
        if len(best_pts) < N:
            # Pad with random points if needed (shouldn't happen)
            while len(best_pts) < N:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.5)
                best_pts = np.vstack([best_pts, [[radius * np.cos(angle), radius * np.sin(angle)]]])
        else:
            best_pts = best_pts[:N]
    
    return best_pts

# EVOLVE-BLOCK-END
