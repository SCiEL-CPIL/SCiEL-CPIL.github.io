# EVOLVE-BLOCK-START
import numpy as np
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon that maximizes the minimum vertex score.
    
    Strategy: Directly optimize 20 vertex coordinates using differential evolution
    (global, derivative-free) on a smooth surrogate, then polish with local search.
    This avoids the Fourier parameterization bottleneck and handles the non-smooth
    min-vertex-score objective more robustly.
    """
    start_time = time.time()
    deadline = 630.0  # safety margin before 630s kill
    
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
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
        
        # Check for repeated points (fast check)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                    return False
        
        # Compute cross products of consecutive edges
        turns = []
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i + 1) % n_pts]
            p2 = pts[(i + 2) % n_pts]
            
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            turns.append(cross)
        
        turns = np.array(turns)
        
        # All turns must be nonzero and of the same sign
        if np.any(np.abs(turns) < 1e-10):
            return False
        
        if np.all(turns > 0) or np.all(turns < 0):
            return True
        
        return False
    
    def vertex_score_fast(v_idx, pts):
        """
        Compute score for vertex v_idx given all points pts.
        Vectorized distance computation.
        """
        n_pts = len(pts)
        v = pts[v_idx]
        
        # Distances to other vertices
        diffs = pts - v
        dists_sq = np.sum(diffs ** 2, axis=1)
        dists_sq[v_idx] = np.inf  # exclude self
        
        # Sort distances (excluding self)
        other_dists = np.sqrt(dists_sq)
        other_dists = np.sort(other_dists[other_dists < 1e15])  # 19 distances
        
        # Check if any distance < 0.05
        if np.any(other_dists < 0.05):
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = other_dists[1:] - other_dists[:-1]
        k = np.argmin(diffs)
        
        D = (other_dists[k] + other_dists[k + 1]) / 2.0
        if D < 0.1:
            return 0.0
        
        # Take four distances
        if k == 0:
            four_dists = other_dists[0:4]
        elif k == 17:
            four_dists = other_dists[15:19]
        else:
            four_dists = other_dists[k - 1:k + 3]
        
        # raw = mean of max(d/D, D/d) over the four distances
        ratios = np.maximum(four_dists / D, D / four_dists)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)
    
    def compute_all_vertex_scores(pts):
        """Compute scores for all vertices."""
        scores = np.array([vertex_score_fast(i, pts) for i in range(len(pts))])
        return scores
    
    def polygon_score(pts):
        """Compute the minimum vertex score over all vertices."""
        scores = compute_all_vertex_scores(pts)
        return np.min(scores)
    
    def smooth_objective(flat_params, temperature=0.05):
        """
        Smooth surrogate for optimization.
        Takes flat 40-dim array of 20 (x,y) pairs.
        Returns negative smooth min score (we minimize this).
        """
        pts = flat_params.reshape(n, 2)
        pts_norm, _ = normalize_polygon(pts)
        
        if not is_strictly_convex(pts_norm):
            return 100.0  # Large penalty for non-convex
        
        scores = compute_all_vertex_scores(pts_norm)
        
        # Soft-min: -T * log(sum(exp(-s_i/T)))
        # Clip to avoid overflow
        scores_clipped = np.clip(scores, 1e-10, 1.0)
        min_score = -temperature * np.log(np.sum(np.exp(-scores_clipped / temperature)))
        
        return -min_score
    
    def exact_objective(flat_params):
        """Exact (non-smooth) objective for final evaluation."""
        pts = flat_params.reshape(n, 2)
        pts_norm, _ = normalize_polygon(pts)
        
        if not is_strictly_convex(pts_norm):
            return -100.0
        
        scores = compute_all_vertex_scores(pts_norm)
        return np.min(scores)
    
    # Initial guess: regular polygon
    x_init = np.cos(base_angles)
    y_init = np.sin(base_angles)
    regular_pts = np.stack([x_init, y_init], axis=1)
    regular_flat = regular_pts.flatten()
    
    best_params = regular_flat.copy()
    best_score = exact_objective(best_params)
    
    # Check time
    if time.time() > deadline - 60.0:
        final_pts = best_params.reshape(n, 2)
        final_pts_norm, _ = normalize_polygon(final_pts)
        if not is_strictly_convex(final_pts_norm):
            final_pts_norm = regular_pts
        return final_pts_norm
    
    # Phase 1: Differential Evolution (global search)
    from scipy.optimize import differential_evolution
    
    # Bounds: coordinates in [-3, 3] should be sufficient after normalization
    bounds = [(-3.0, 3.0)] * (2 * n)
    
    try:
        de_result = differential_evolution(
            smooth_objective,
            bounds=bounds,
            x0=regular_flat,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.5),
            recombination=0.8,
            seed=42,
            polish=False,  # We'll do our own polish
            updateinterval=10,
            callback=None
        )
        
        if de_result.fun < smooth_objective(best_params):
            test_score = exact_objective(de_result.x)
            if test_score > best_score:
                best_score = test_score
                best_params = de_result.x.copy()
        
        # Also check the initial point
        init_score = exact_objective(regular_flat)
        if init_score > best_score:
            best_score = init_score
            best_params = regular_flat.copy()
    
    except Exception as e:
        pass  # Continue with regular polygon if DE fails
    
    if time.time() > deadline - 30.0:
        final_pts = best_params.reshape(n, 2)
        final_pts_norm, _ = normalize_polygon(final_pts)
        if not is_strictly_convex(final_pts_norm):
            final_pts_norm = regular_pts
        return final_pts_norm
    
    # Phase 2: Local polish with Nelder-Mead on the exact objective
    from scipy.optimize import minimize
    
    def negative_exact_objective(flat_params):
        """Negative of exact score for minimization."""
        return -exact_objective(flat_params)
    
    try:
        local_result = minimize(
            negative_exact_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        local_score = exact_objective(local_result.x)
        if local_score > best_score:
            best_score = local_score
            best_params = local_result.x.copy()
    
    except Exception as e:
        pass
    
    if time.time() > deadline - 10.0:
        final_pts = best_params.reshape(n, 2)
        final_pts_norm, _ = normalize_polygon(final_pts)
        if not is_strictly_convex(final_pts_norm):
            final_pts_norm = regular_pts
        return final_pts_norm
    
    # Phase 3: Additional random restarts with local optimization
    np.random.seed(123)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Random perturbation of the regular polygon
        angle_perturb = np.random.normal(0.0, 0.1, size=n)
        radius_perturb = 1.0 + np.random.normal(0.0, 0.05, size=n)
        
        angles = base_angles + angle_perturb
        radii = radius_perturb
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        flat = pts.flatten()
        
        # Local optimization from this starting point
        try:
            result = minimize(
                negative_exact_objective,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            score = exact_objective(result.x)
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
                
        except Exception as e:
            pass
    
    # Final result
    final_pts = best_params.reshape(n, 2)
    final_pts_norm, _ = normalize_polygon(final_pts)
    
    # Ensure strict convexity; if not, fall back to regular polygon
    if not is_strictly_convex(final_pts_norm):
        final_pts_norm = regular_pts
    
    return final_pts_norm


# EVOLVE-BLOCK-END
