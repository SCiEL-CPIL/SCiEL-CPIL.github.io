import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 135.0  # Safety margin for 150s limit

    def score(points):
        """Compute the score: product of distances after scaling to diameter 2, divided by n^n."""
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    def regular_octagon():
        """Return regular octagon inscribed in unit circle (diameter 2)."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    best_score = score(regular_octagon())
    best_points = regular_octagon().copy()

    # ============================================================
    # STRATEGY 1: Convex polygon via edge lengths
    # Parametrize 8 points as vertices of a convex polygon defined by
    # 8 positive edge lengths summing to a fixed perimeter.
    # This inherently enforces convexity and distinctness.
    # ============================================================
    
    def build_convex_polygon(edge_lengths):
        """
        Given 8 edge lengths (positive, summing to perimeter),
        construct the convex polygon with those edge lengths.
        The polygon is determined up to rotation/translation.
        We place it so that the centroid is at origin and compute
        the actual vertex positions.
        
        Key insight: For a given set of edge lengths, there's a unique
        (up to congruence) convex polygon. We can construct it by
        finding the angles that close the polygon.
        
        Simpler approach: Use the fact that for a convex polygon with
        given side lengths, we can find the circumradius and angles.
        But this is complex. Instead, let's use a different approach:
        
        Alternative: Parametrize by angles between consecutive edges.
        For a closed convex polygon, the turning angles sum to 2*pi.
        Each turning angle is in (0, pi) for strict convexity.
        """
        # Use turning angles parametrization
        # 7 turning angles (the 8th is determined by closure)
        # Actually, for a closed polygon, sum of exterior angles = 2*pi
        # So we need 7 free turning angles, each in (0, pi)
        # But we also need the edge lengths.
        
        # Let's use a different approach: optimize the 8 angles directly
        # where angle_i is the direction of edge i, and we require
        # the polygon to close.
        
        # Actually, let me use a cleaner approach:
        # Parametrize by 7 turning angles t_1,...,t_7 in (0, pi)
        # t_8 = 2*pi - sum(t_1,...,t_7) must also be in (0, pi)
        # Edge lengths are given.
        
        # This is getting complex. Let me use a simpler structural change:
        # Optimize in 2D with a different parameterization that avoids
        # the symmetry assumption.
        
        # NEW APPROACH: Use polar coordinates around the centroid
        # r_i, theta_i for i=0..7, with constraints that points are
        # in convex position.
        
        # Actually, the simplest robust approach: just do full 2D optimization
        # with multiple random restarts and a better global optimizer.
        
        # For now, return None to signal this strategy needs rework
        return None

    # ============================================================
    # STRATEGY 2: Full 2D optimization with dual_annealing
    # This explores the full 16D space without symmetry assumptions
    # ============================================================
    
    def full_2d_objective(params):
        """
        params: 16 floats representing 8 points (x0,y0,x1,y1,...,x7,y7)
        Returns negative log of product of distances (to minimize).
        """
        pts = params.reshape(8, 2)
        
        # Check for coincident points
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_d = np.min(dists[np.triu(np.ones((8,8), dtype=bool), k=1)])
        if min_d < 1e-6:
            return 1e10
        
        # Scale to diameter 2
        D = np.max(dists)
        if D < 1e-6:
            return 1e10
        
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
        unique_dists = scaled_dists[mask]
        
        # Add small epsilon to avoid log(0)
        log_sum = np.sum(np.log(np.maximum(unique_dists, 1e-10)))
        return -log_sum

    # Bounds: keep points in a reasonable range
    bounds_full = [(-2.0, 2.0), (-2.0, 2.0)] * 8

    # Try dual_annealing for global optimization
    if time.time() < deadline - 30:
        try:
            da_res = dual_annealing(
                full_2d_objective,
                bounds=bounds_full,
                seed=42,
                maxiter=100,
                initial_temp=5230.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                clip=0.5,
                no_local_search=False,
                maxiter_no_improvement=20
            )
            
            if da_res.success:
                pts = da_res.x.reshape(8, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    print(f"Dual annealing found: {s:.6f}")
        except Exception as e:
            print(f"Dual annealing error: {e}")

    # ============================================================
    # STRATEGY 3: Multiple random restarts with L-BFGS-B
    # ============================================================
    
    if time.time() < deadline - 15:
        np.random.seed(123)
        n_restarts = 30
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate random initial points
            # Use different distributions to explore different regions
            if r % 3 == 0:
                # Random points in unit disk
                angles = np.random.uniform(0, 2*np.pi, 8)
                radii = np.random.uniform(0.3, 1.0, 8)
                pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            elif r % 3 == 1:
                # Perturbed regular octagon
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                angles += np.random.uniform(-0.3, 0.3, 8)
                radii = np.random.uniform(0.7, 1.0, 8)
                pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            else:
                # Two clusters
                cluster1_angles = np.random.uniform(0, np.pi/2, 4)
                cluster2_angles = np.random.uniform(3*np.pi/2, 2*np.pi, 4)
                angles = np.concatenate([cluster1_angles, cluster2_angles])
                radii = np.random.uniform(0.5, 1.0, 8)
                pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            params_init = pts_init.flatten()
            
            try:
                res = minimize(
                    full_2d_objective,
                    params_init,
                    method='L-BFGS-B',
                    bounds=bounds_full,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success:
                    pts = res.x.reshape(8, 2)
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
                        print(f"Restart {r} found: {s:.6f}")
            except Exception:
                pass

    # ============================================================
    # STRATEGY 4: Differential evolution in full 2D space
    # ============================================================
    
    if time.time() < deadline - 10:
        try:
            de_res = differential_evolution(
                full_2d_objective,
                bounds=bounds_full,
                maxiter=50,
                tol=1e-6,
                seed=77,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.8
            )
            if de_res.success:
                pts = de_res.x.reshape(8, 2)
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                    print(f"DE found: {s:.6f}")
        except Exception as e:
            print(f"DE error: {e}")

    # ============================================================
    # STRATEGY 5: Final polish with Nelder-Mead on best found
    # ============================================================
    
    if time.time() < deadline - 3:
        try:
            params_best = best_points.flatten()
            res = minimize(
                full_2d_objective,
                params_best,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            pts = res.x.reshape(8, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                print(f"Polish found: {s:.6f}")
        except Exception:
            pass

    # Ensure points are distinct and within bounds
    final_pts = best_points
    # Scale to ensure |coordinate| < 10^6 (should already be satisfied)
    max_coord = np.max(np.abs(final_pts))
    if max_coord > 1e6:
        final_pts = final_pts * (1e6 / max_coord)
    
    # Verify distinctness
    diffs = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
    min_dist = np.min(np.linalg.norm(diffs, axis=2)[np.triu(np.ones((8,8), dtype=bool), k=1)])
    if min_dist < 1e-8:
        # Fallback to regular octagon
        final_pts = regular_octagon()

    return final_pts.tolist()
