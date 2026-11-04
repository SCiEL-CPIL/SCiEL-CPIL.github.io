import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Each inner hexagon may translate and rotate.
    """
    # Time budget guard
    import time
    start_time = time.time()
    TIME_LIMIT = 140.0  # 150s wall clock, 10s safety margin
    
    # Standard hexagonal lattice cluster: 1 center + 6 ring1 + 4 ring2
    # Ring 1: 6 neighbors at distance sqrt(3)
    # Ring 2: 4 at distance 3, angles 30, 90, 150, 210
    base_centers = np.array([
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)*0.5, np.sqrt(3)*0.5*3],  # 60 deg
        [-np.sqrt(3)*0.5, np.sqrt(3)*0.5*3],  # 120 deg
        [-np.sqrt(3), 0.0],                   # 180 deg
        [-np.sqrt(3)*0.5, -np.sqrt(3)*0.5*3], # 240 deg
        [np.sqrt(3)*0.5, -np.sqrt(3)*0.5*3],  # 300 deg
        [3.0*np.cos(np.deg2rad(30)), 3.0*np.sin(np.deg2rad(30))],
        [3.0*np.cos(np.deg2rad(90)), 3.0*np.sin(np.deg2rad(90))],
        [3.0*np.cos(np.deg2rad(150)), 3.0*np.sin(np.deg2rad(150))],
        [3.0*np.cos(np.deg2rad(210)), 3.0*np.sin(np.deg2rad(210))],
    ])
    
    # All hexagons initially at 30 degrees rotation
    base_angles = np.full(11, 30.0)
    
    # Support function of a regular hexagon with side s, rotation theta
    # For a point p, the support function h(p) = max over vertices v of (p·v)
    # A regular hexagon centered at origin with side s, rotation theta has vertices:
    # v_k = s * (cos(theta + k*60), sin(theta + k*60)) for k=0..5
    # Actually for a regular hexagon with side length s, the circumradius is s
    # So vertices are at distance s from center.
    
    def hex_vertices(cx, cy, s, theta_deg):
        """Return 6 vertices of regular hexagon centered at (cx,cy), side s, rotation theta_deg."""
        theta = np.deg2rad(theta_deg)
        angles = theta + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def hex_support(p, cx, cy, s, theta_deg):
        """
        Compute max dot product of p with vertices of hexagon centered at (cx,cy).
        This gives the support function value in direction p.
        """
        theta = np.deg2rad(theta_deg)
        angles = theta + np.deg2rad(np.arange(6) * 60)
        vx = s * np.cos(angles)
        vy = s * np.sin(angles)
        # p is a unit vector (px, py)
        return np.max(p[0] * (cx + vx) + p[1] * (cy + vy))
    
    def outer_hex_side_from_points(points, outer_theta_deg):
        """
        Given a set of points (vertices of all inner hexagons), find the minimum
        side length s of a regular hexagon with rotation outer_theta_deg that contains all points.
        
        For a regular hexagon with side s and rotation phi, the containment constraint is:
        For each of 6 directions (normal to each edge), the projection of all points 
        onto that direction must be within [-s*sqrt(3)/2, s*sqrt(3)/2].
        
        The edge normals are at angles phi + k*60 for k=0..5.
        The distance from center to each edge is s*sqrt(3)/2.
        """
        phi = np.deg2rad(outer_theta_deg)
        normals = phi + np.deg2rad(np.arange(6) * 60)
        nx = np.cos(normals)
        ny = np.sin(normals)
        
        # Project all points onto each normal direction
        # points: (N, 2)
        proj = points[:, 0] * nx[None, :] + points[:, 1] * ny[None, :]
        # For each normal, find max and min projection
        max_proj = np.max(proj, axis=0)
        min_proj = np.min(proj, axis=0)
        
        # The hexagon centered at origin with edge distance d = s*sqrt(3)/2
        # requires: max_proj <= d and -min_proj <= d for all 6 normals
        # So d >= max(max_proj, -min_proj)
        # And s = d / (sqrt(3)/2) = 2*d/sqrt(3)
        d = np.max(np.maximum(max_proj, -min_proj))
        return 2.0 * d / np.sqrt(3)
    
    def get_all_vertices(centers, angles_deg, s_inner=1.0):
        """Get all 66 vertices (11 hexagons * 6 vertices)."""
        all_verts = []
        for i in range(len(centers)):
            cx, cy = centers[i]
            theta = angles_deg[i]
            angles = np.deg2rad(theta) + np.deg2rad(np.arange(6) * 60)
            vx = s_inner * np.cos(angles)
            vy = s_inner * np.sin(angles)
            all_verts.append(np.column_stack([cx + vx, cy + vy]))
        return np.vstack(all_verts)
    
    def objective(params):
        """
        params: 11*3 = 33 values: for each hex, (dx, dy, dtheta_deg)
        Returns the outer hexagon side length.
        """
        if time.time() - start_time > TIME_LIMIT:
            return 1e10  # timeout penalty
        
        centers = base_centers.copy()
        angles = base_angles.copy()
        for i in range(11):
            centers[i, 0] += params[3*i]
            centers[i, 1] += params[3*i+1]
            angles[i] += params[3*i+2]
        
        # Get all vertices
        verts = get_all_vertices(centers, angles)
        
        # Optimize outer hexagon rotation to minimize side length
        # Use a few candidate rotations
        best_side = np.inf
        for outer_theta in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]:
            s = outer_hex_side_from_points(verts, outer_theta)
            if s < best_side:
                best_side = s
        
        return best_side
    
    def objective_smooth(params, beta=50.0):
        """
        Smooth version using log-sum-exp for the max over normals.
        """
        if time.time() - start_time > TIME_LIMIT:
            return 1e10
        
        centers = base_centers.copy()
        angles = base_angles.copy()
        for i in range(11):
            centers[i, 0] += params[3*i]
            centers[i, 1] += params[3*i+1]
            angles[i] += params[3*i+2]
        
        verts = get_all_vertices(centers, angles)
        
        best_side = np.inf
        for outer_theta in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            phi = np.deg2rad(outer_theta)
            normals = phi + np.deg2rad(np.arange(6) * 60)
            nx = np.cos(normals)
            ny = np.sin(normals)
            
            proj = verts[:, 0] * nx[None, :] + verts[:, 1] * ny[None, :]
            
            # Soft max over points for each normal
            # For max: log(sum(exp(beta*x))) / beta
            # For min: -log(sum(exp(-beta*x))) / beta
            max_proj = np.log(np.sum(np.exp(beta * proj), axis=0)) / beta
            min_proj = -np.log(np.sum(np.exp(-beta * proj), axis=0)) / beta
            
            d_vals = np.maximum(max_proj, -min_proj)
            # Soft max over the 6 normals
            d = np.log(np.sum(np.exp(beta * d_vals))) / beta
            s = 2.0 * d / np.sqrt(3)
            
            if s < best_side:
                best_side = s
        
        return best_side
    
    # Try multiple initial perturbations
    best_result = None
    best_side = np.inf
    
    # Initial guess: standard packing
    x0 = np.zeros(33)
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_side:
            best_side = res.fun
            best_result = res.x
    except Exception:
        pass
    
    if time.time() - start_time > TIME_LIMIT - 20:
        # Not enough time for more, return current best
        pass
    else:
        # Try with smooth objective
        try:
            res2 = minimize(objective_smooth, x0, method='BFGS', 
                          options={'maxiter': 2000, 'gtol': 1e-12})
            if res2.fun < best_side:
                best_side = res2.fun
                best_result = res2.x
        except Exception:
            pass
    
    if time.time() - start_time > TIME_LIMIT - 10:
        pass
    else:
        # Try small random perturbations as starting points
        np.random.seed(42)
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            x0_pert = np.random.normal(0, 0.05, 33)
            try:
                res3 = minimize(objective, x0_pert, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                if res3.fun < best_side:
                    best_side = res3.fun
                    best_result = res3.x
            except Exception:
                pass
    
    # If optimization didn't find anything better, use the original
    if best_result is None:
        best_result = np.zeros(33)
    
    # Build final inner hexagons
    centers = base_centers.copy()
    angles = base_angles.copy()
    for i in range(11):
        centers[i, 0] += best_result[3*i]
        centers[i, 1] += best_result[3*i+1]
        angles[i] += best_result[3*i+2]
    
    inner = []
    for i in range(11):
        inner.append([centers[i, 0], centers[i, 1], angles[i]])
    
    # Determine optimal outer hexagon rotation and side
    verts = get_all_vertices(centers, angles)
    best_s = np.inf
    best_outer_theta = 0.0
    for outer_theta in np.arange(0, 180, 1.0):
        s = outer_hex_side_from_points(verts, outer_theta)
        if s < best_s:
            best_s = s
            best_outer_theta = outer_theta
    
    # Refine outer theta with finer grid
    for outer_theta in np.arange(best_outer_theta - 1, best_outer_theta + 1, 0.01):
        s = outer_hex_side_from_points(verts, outer_theta)
        if s < best_s:
            best_s = s
            best_outer_theta = outer_theta
    
    # Add small epsilon for safety
    eps = 1e-9
    side = best_s + eps
    
    return inner, (0.0, 0.0), side, best_outer_theta
