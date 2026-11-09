# EVOLVE-BLOCK-START
import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex score.
    
    Strategy: Start from regular polygon, parameterize angular deviations using
    low-frequency Fourier modes (k=1,2,3) to maintain global smoothness.
    Optimize the parameters to maximize the minimum vertex score using a
    smooth surrogate (soft-min) with L-BFGS-B, then polish with a high-precision
    local solve.
    """
    start_time = time.time()
    deadline = 600.0
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Fourier modes: k=1, 2, 3 (skip k=0 which would be uniform rotation, k=n/2=10 which is degenerate)
    # For each mode k, we have amplitude and phase: 2 parameters per mode
    # Total: 3 modes * 2 = 6 parameters
    # But we also allow radius modulation: another set of 6 parameters
    # Let's use: for each k in [1,2,3], angular deviation = a_k * cos(k*theta + phi_k)
    # and radius modulation = r_k * cos(k*theta + psi_k)
    # Total: 12 parameters
    
    def params_to_polygon(params):
        """Convert 12 parameters to 20 vertex coordinates."""
        # params[0:6] = [a1, phi1, a2, phi2, a3, phi3] for angular deviations
        # params[6:12] = [r1, psi1, r2, psi2, r3, psi3] for radius modulation
        
        a1, phi1, a2, phi2, a3, phi3 = params[0:6]
        r1, psi1, r2, psi2, r3, psi3 = params[6:12]
        
        # Angular deviations
        theta_dev = (a1 * np.cos(1 * base_angles + phi1) +
                     a2 * np.cos(2 * base_angles + phi2) +
                     a3 * np.cos(3 * base_angles + phi3))
        
        # Radius modulation
        radius_mod = (1.0 + 
                      r1 * np.cos(1 * base_angles + psi1) +
                      r2 * np.cos(2 * base_angles + psi2) +
                      r3 * np.cos(3 * base_angles + psi3))
        
        # Ensure radius stays positive
        radius_mod = np.maximum(radius_mod, 0.5)
        
        angles = base_angles + theta_dev
        radii = radius_mod
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        return np.stack([x, y], axis=1)
    
    def normalize_polygon(pts):
        """Normalize: subtract centroid, divide by mean distance to centroid."""
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        dists = np.linalg.norm(pts_centered, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return pts_centered, mean_dist
        return pts_centered / mean_dist, mean_dist
    
    def is_strictly_convex(pts):
        """Check if polygon is strictly convex with consistent turning direction."""
        n_pts = len(pts)
        if n_pts < 3:
            return False
        
        # Check for repeated points
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    return False
        
        # Compute cross products of consecutive edges
        turns = []
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i+1) % n_pts]
            p2 = pts[(i+2) % n_pts]
            
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            turns.append(cross)
        
        turns = np.array(turns)
        
        # All turns must be nonzero and of the same sign
        if np.any(np.abs(turns) < 1e-12):
            return False
        
        # All same sign?
        if np.all(turns > 0) or np.all(turns < 0):
            # Check total turning is ±2π
            # For a convex polygon, total turning should be 2π (or -2π)
            # This is automatically satisfied if all turns are same sign for a simple closed curve
            return True
        
        return False
    
    def vertex_score(v, pts):
        """
        Compute score for vertex v given all points pts.
        """
        n_pts = len(pts)
        idx = np.where(np.all(pts == v, axis=1))[0]
        if len(idx) == 0:
            return 0.0
        idx = idx[0]
        
        # Distances to other vertices
        dists = []
        for j in range(n_pts):
            if j != idx:
                dists.append(np.linalg.norm(pts[idx] - pts[j]))
        
        dists = np.sort(np.array(dists))  # d_0 <= ... <= d_18
        
        # Check if any distance < 0.05
        if np.any(dists < 0.05):
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = dists[1:] - dists[:-1]
        k = np.argmin(diffs)
        
        D = (dists[k] + dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        
        # Take four distances
        if k == 0:
            four_dists = dists[0:4]
        elif k == 17:
            four_dists = dists[15:19]
        else:
            four_dists = dists[k-1:k+3]
        
        # raw = mean of max(d/D, D/d) over the four distances
        ratios = np.maximum(four_dists / D, D / four_dists)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)
    
    def polygon_score(pts):
        """Compute the minimum vertex score over all vertices."""
        n_pts = len(pts)
        scores = []
        for i in range(n_pts):
            s = vertex_score(pts[i], pts)
            scores.append(s)
        return min(scores)
    
    def smooth_polygon_score(pts, temperature=0.01):
        """
        Smooth surrogate using soft-min.
        """
        n_pts = len(pts)
        scores = np.array([vertex_score(pts[i], pts) for i in range(n_pts)])
        
        # Soft-min: -T * log(sum(exp(-s_i/T)))
        # For T -> 0, this approaches min(s_i)
        min_score = -temperature * np.log(np.sum(np.exp(-scores / temperature)))
        return min_score
    
    def objective(params):
        """Negative of smooth polygon score (we minimize this)."""
        pts = params_to_polygon(params)
        pts_norm, _ = normalize_polygon(pts)
        
        if not is_strictly_convex(pts_norm):
            return 1000.0  # Large penalty for non-convex
        
        return -smooth_polygon_score(pts_norm, temperature=0.01)
    
    # Initial parameters: all zeros (regular polygon)
    params0 = np.zeros(12)
    
    # Try multiple random initializations
    best_params = params0.copy()
    best_score = -1e10
    
    # First, evaluate the regular polygon
    pts0 = params_to_polygon(params0)
    pts0_norm, _ = normalize_polygon(pts0)
    if is_strictly_convex(pts0_norm):
        best_score = polygon_score(pts0_norm)
    
    # L-BFGS-B optimization
    from scipy.optimize import minimize
    
    n_random_starts = 50
    max_iter_per_start = 200
    
    for trial in range(n_random_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Random initial parameters with small magnitudes
        np.random.seed(trial)
        params_init = np.random.normal(0.0, 0.1, size=12)
        
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter_per_start, 'ftol': 1e-8}
        )
        
        if result.success or result.fun < objective(best_params):
            test_pts = params_to_polygon(result.x)
            test_pts_norm, _ = normalize_polygon(test_pts)
            if is_strictly_convex(test_pts_norm):
                test_score = polygon_score(test_pts_norm)
                if test_score > best_score:
                    best_score = test_score
                    best_params = result.x.copy()
        
        # Also try the initial point
        init_pts = params_to_polygon(params_init)
        init_pts_norm, _ = normalize_polygon(init_pts)
        if is_strictly_convex(init_pts_norm):
            init_score = polygon_score(init_pts_norm)
            if init_score > best_score:
                best_score = init_score
                best_params = params_init.copy()
    
    # Polish: use Nelder-Mead with tighter tolerance
    if time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        
        final_result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        
        final_pts = params_to_polygon(final_result.x)
        final_pts_norm, _ = normalize_polygon(final_pts)
        if is_strictly_convex(final_pts_norm):
            final_score = polygon_score(final_pts_norm)
            if final_score > best_score:
                best_score = final_score
                best_params = final_result.x.copy()
    
    # Get final polygon
    final_pts = params_to_polygon(best_params)
    final_pts_norm, _ = normalize_polygon(final_pts)
    
    # Ensure strict convexity; if not, fall back to regular polygon
    if not is_strictly_convex(final_pts_norm):
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        final_pts_norm = np.stack([x, y], axis=1)
    
    return final_pts_norm


# EVOLVE-BLOCK-END
