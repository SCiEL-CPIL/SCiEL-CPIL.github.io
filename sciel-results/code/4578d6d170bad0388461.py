"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    def get_covering_info(pts):
        """
        Computes the convex hull, finds the facet with the largest circumradius,
        and returns the circumcenter of that facet (the worst-case point on the sphere).
        Also returns the covering radius (angular distance).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            tree = cKDTree(pts)
            grid_lat = np.linspace(-np.pi/2, np.pi/2, 50)
            grid_lon = np.linspace(0, 2*np.pi, 100)
            lat, lon = np.meshgrid(grid_lat, grid_lon)
            gp = np.column_stack([
                (np.cos(lat)*np.cos(lon)).ravel(),
                (np.cos(lat)*np.sin(lon)).ravel(),
                (np.sin(lat)).ravel()
            ])
            dists, _ = tree.query(gp)
            angles = 2.0 * np.arcsin(np.clip(dists/2.0, 0, 1))
            return np.max(angles), gp[np.argmax(angles)]

        max_radius = 0
        worst_center = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
                
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                
        if worst_center is None:
            return 0.0, np.random.randn(3) / np.linalg.norm(np.random.randn(3))
            
        return max_radius, worst_center

    def calc_covering_radius(pts):
        radius, _ = get_covering_info(pts)
        return radius

    def fast_radius_estimate(pts):
        """Fast approximate covering radius using k-d tree on a sparse grid."""
        tree = cKDTree(pts)
        # Use a coarser grid for speed
        grid_lat = np.linspace(-np.pi/2, np.pi/2, 30)
        grid_lon = np.linspace(0, 2*np.pi, 60)
        lat, lon = np.meshgrid(grid_lat, grid_lon)
        gp = np.column_stack([
            (np.cos(lat)*np.cos(lon)).ravel(),
            (np.cos(lat)*np.sin(lon)).ravel(),
            (np.sin(lat)).ravel()
        ])
        dists, _ = tree.query(gp)
        angles = 2.0 * np.arcsin(np.clip(dists/2.0, 0, 1))
        return np.max(angles)

    def repulsive_force(pts):
        """Compute repulsive forces between all pairs of points."""
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-10:
                    continue
                # Inverse square law repulsion, but capped to avoid explosion
                force = diff / (dist_sq * np.sqrt(dist_sq))
                # Cap the force magnitude
                force_norm = np.linalg.norm(force)
                if force_norm > 1.0:
                    force /= force_norm
                forces[i] += force
        return forces

    def optimize_single(pts_init, max_iters=300):
        """Optimize a single starting configuration."""
        current = pts_init.copy()
        best = current.copy()
        best_r = calc_covering_radius(current)
        
        step_size = 0.05
        decay = 0.995
        repulse_weight = 0.001
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            radius, worst_center = get_covering_info(current)
            
            if radius < best_r:
                best_r = radius
                best = current.copy()
            
            # Find the worst facet
            try:
                hull = ConvexHull(current)
            except:
                break
            
            target_indices = None
            for simplex in hull.simplices:
                v1 = current[simplex[0]]
                v2 = current[simplex[1]]
                v3 = current[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break
            
            if target_indices is None:
                break
            
            # Compute forces
            forces = np.zeros_like(current)
            
            # Attraction: move worst facet vertices toward worst_center
            for idx in target_indices:
                v = current[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    forces[idx] += step_size * dir_vec / norm_dir
            
            # Repulsion: all points repel each other
            forces += repulse_weight * repulsive_force(current)
            
            # Apply forces
            current += forces
            current /= np.linalg.norm(current, axis=1, keepdims=True)
            
            step_size *= decay
            repulse_weight *= 0.999
            
            if step_size < 0.0005:
                # Perturb and restart
                current += np.random.randn(N, 3) * 0.02
                current /= np.linalg.norm(current, axis=1, keepdims=True)
                step_size = 0.03
                repulse_weight = 0.001
        
        return best, best_r

    # Multi-start optimization
    best_points = None
    best_radius = np.pi  # Max possible radius
    
    # Start 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [fib_points]
    
    # Add random starts
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.randn(N, 3)
        rand_pts /= np.linalg.norm(rand_pts, axis=1, keepdims=True)
        starts.append(rand_pts)
    
    for start_idx, start_pts in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        # Run optimization
        pts, rad = optimize_single(start_pts)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Final L-BFGS polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        def objective_flat(x):
            pts = x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            return calc_covering_radius(pts)
        
        x0 = best_points.ravel()
        
        # Use L-BFGS with line search
        result = minimize(objective_flat, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'gtol': 1e-6})
        
        if result.fun < best_radius:
            best_points = result.x.reshape(N, 3)
            best_radius = result.fun
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
