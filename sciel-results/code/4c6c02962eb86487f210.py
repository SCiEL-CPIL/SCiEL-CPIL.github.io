import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Objective function to maximize: (d_min / d_max)^2
    # We will minimize the negative of this.
    def objective(points):
        # points is (14, 3)
        # Ensure points are on unit sphere? 
        # The optimizer might drift off. 
        # We should constrain or project.
        # Nelder-Mead on spherical coords is better to enforce constraint.
        # But for repulsion, we project manually.
        
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def repulsion_step(points, eta):
        # points: (n, 3)
        # Vectorized force calculation
        # dists matrix (n, n)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        dists_sq[np.diag_indices_from(dists_sq)] = 1.0
        dists = np.sqrt(dists_sq)
        
        # Force magnitude: 1/d^2
        # F_ij = (p_i - p_j) / d^3  (since vector is diff, magnitude 1/d, so 1/d^2 * 1/d = 1/d^3)
        # Wait, F = k/d^2 * unit_vec. unit_vec = diff/d. So F = k * diff / d^3.
        # k=1.
        inv_d3 = np.zeros_like(dists_sq)
        inv_d3 = 1.0 / (dists_sq * dists)
        inv_d3[np.diag_indices_from(inv_d3)] = 0.0
        
        forces = diff * inv_d3[:, :, np.newaxis]
        
        # Sum forces for each point i
        # forces[i, j] is force on i from j?
        # diff = p_i - p_j. So forces[i, j] is vector from j to i scaled.
        # We want net force on i. Sum over j.
        net_forces = np.sum(forces, axis=1)
        
        # Project onto tangent plane
        # p_i is normal.
        # f_tan = f - (f . p_i) p_i
        dot = np.sum(net_forces * points, axis=1)
        net_forces_tangent = net_forces - dot[:, np.newaxis] * points
        
        # Update
        new_points = points + eta * net_forces_tangent
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        # Avoid div by 0
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        return new_points

    best_score = -1
    best_points = None
    
    # Multi-start
    np.random.seed(42)
    
    for start in range(20):
        # Random initialization on sphere
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Repulsion simulation
        eta = 0.1
        for i in range(1000):
            pts = repulsion_step(pts, eta)
            eta *= 0.999 # Slow decay
            
        # Evaluate
        score = objective(pts)
        
        # Polish with Nelder-Mead on spherical coordinates
        # Convert to spherical (theta, phi)
        # x = r sin(phi) cos(theta)
        # y = r sin(phi) sin(theta)
        # z = r cos(phi)
        # r=1
        # theta = arctan2(y, x)
        # phi = arccos(z)
        
        z = pts[:, 2]
        y = pts[:, 1]
        x = pts[:, 0]
        
        # Clip z for arccos
        z = np.clip(z, -1.0, 1.0)
        phi = np.arccos(z)
        theta = np.arctan2(y, x)
        
        init_sph = np.hstack([theta, phi]).flatten()
        
        def sph_objective(sph_params):
            theta = sph_params[:n]
            phi = sph_params[n:]
            
            # Ensure phi in [0, pi]
            # theta in [0, 2pi]
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            p = np.column_stack((x, y, z))
            
            # Normalize just in case
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            
            return -objective(p) # Minimize negative
        
        try:
            res = minimize(sph_objective, init_sph, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 0: # res.fun is negative score
                final_sph = res.x
                theta = final_sph[:n]
                phi = final_sph[n:]
                
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                
                final_pts = np.column_stack((x, y, z))
                final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)
                
                final_score = objective(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
                    
        except Exception as e:
            pass
            
        if best_score > 0.2: # Heuristic: if we found a good one, maybe stop early? No, keep searching for global.
            pass

    if best_points is None:
        # Fallback
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    return best_points
