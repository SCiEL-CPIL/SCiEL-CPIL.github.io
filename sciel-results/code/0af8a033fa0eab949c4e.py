import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Time management
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for safety
    
    # Objective function to minimize (we minimize the negative of the metric)
    def objective(x):
        # Check time
        if time.time() > deadline:
            return np.inf
            
        pts = x.reshape((n, d))
        
        # Center the points to remove translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate pairwise distances
        try:
            dists = pdist(pts)
        except Exception:
            return np.inf
            
        if np.any(~np.isfinite(dists)):
            return np.inf
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return np.inf
            
        # We want to maximize (d_min / d_max)^2
        # So we minimize -(d_min / d_max)^2
        ratio = (d_min / d_max) ** 2
        return -ratio

    def icosahedron():
        """Return 12 vertices of regular icosahedron on unit sphere."""
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        # Normalize to unit sphere
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def generate_symmetric_seeds():
        """
        Generate high-quality symmetric seeds.
        Strategy: 12 points form an icosahedron. 2 points are placed symmetrically.
        Possible symmetric placements for the extra 2 points:
        1. Along the 5-fold axis (z-axis), at +/- z.
        2. Along a 3-fold axis.
        3. Along a 2-fold axis.
        4. The 2 points themselves form a dipole aligned with an axis of symmetry.
        """
        ico = icosahedron()
        seeds = []
        
        # Case 1: Extra points on the z-axis (5-fold axis of icosahedron)
        # The icosahedron has a 5-fold axis passing through two opposite vertices.
        # Let's align vertex 0 to be the "top" pole.
        # Vertex 0 is [0, 1, phi] normalized. Let's rotate so it aligns with Z.
        v0 = ico[0]
        v0_norm = np.linalg.norm(v0)
        v0 = v0 / v0_norm
        
        # Rotation matrix to align v0 with [0,0,1]
        # Cross product v0 x z gives axis, dot gives angle
        z = np.array([0, 0, 1])
        axis = np.cross(v0, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(v0, z))
            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rotated = ico @ R.T
        else:
            ico_rotated = ico.copy()
            
        # Now ico_rotated[0] is at [0,0,1] and ico_rotated[2] (opposite) is at [0,0,-1]
        # Actually, let's verify which one is opposite.
        # In standard icosahedron coords, [0, 1, phi] and [0, -1, -phi] are opposite.
        # So ico_rotated[0] is North Pole, ico_rotated[2] is South Pole.
        
        # Try different radii for the extra points on the Z-axis
        for r_extra in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 2: Extra points on a 3-fold axis
        # A 3-fold axis passes through the centers of two opposite triangular faces.
        # The centroid of face (0, 1, 10) is a 3-fold axis direction.
        # Let's pick a face. Vertices 0, 1, 10 are connected?
        # 0: [0, 1, phi]
        # 1: [0, -1, phi]
        # 10: [phi, 0, 1]
        # Distance 0-1: sqrt(0+4+0)=2. 0-10: sqrt(phi^2+1+1-phi^2) = sqrt(2). 
        # Wait, edge length is 2.
        # Let's just use the centroid of vertices 0, 1, 10.
        c = (ico[0] + ico[1] + ico[10]) / 3.0
        c_norm = np.linalg.norm(c)
        c = c / c_norm
        
        # Rotate this direction to Z
        z = np.array([0, 0, 1])
        axis = np.cross(c, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(c, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rotated_3 = ico @ R.T
        else:
            ico_rotated_3 = ico.copy()
            
        for r_extra in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated_3, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 3: Extra points on a 2-fold axis
        # 2-fold axis passes through midpoints of opposite edges.
        # Midpoint of edge (0, 1) is [0, 0, phi].
        m = (ico[0] + ico[1]) / 2.0
        m_norm = np.linalg.norm(m)
        m = m / m_norm
        
        # Rotate this direction to Z
        z = np.array([0, 0, 1])
        axis = np.cross(m, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(m, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rotated_2 = ico @ R.T
        else:
            ico_rotated_2 = ico.copy()
            
        for r_extra in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated_2, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 4: Random but centered on icosahedron
        np.random.seed(42)
        for _ in range(10):
            p1 = np.random.randn(3)
            p1 /= np.linalg.norm(p1)
            p2 = -p1
            seed = np.vstack([ico, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)

        return seeds

    best_x = None
    best_val = np.inf
    
    seeds = generate_symmetric_seeds()
    
    # Evaluate seeds and pick the best few to refine
    seed_scores = []
    for i, seed in enumerate(seeds):
        x = seed.flatten()
        val = objective(x)
        seed_scores.append((val, i))
        
    # Sort by score (best is lowest value, which is most negative)
    seed_scores.sort()
    
    # Refine the top 5 seeds using local optimization
    for rank, (val, idx) in enumerate(seed_scores[:5]):
        if time.time() > deadline - 60:
            break
            
        x0 = seeds[idx].flatten()
        
        # Use L-BFGS-B for local refinement
        # Note: The objective function has a kink at the minimum distance, 
        # but L-BFGS-B with finite differences can still work well if we are close.
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
        # Also try Nelder-Mead as a backup for non-smoothness
        if time.time() > deadline - 30:
            break
        try:
            res_nm = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

    # If local refinement from symmetric seeds didn't work well or we have time, 
    # run a constrained DE for a short time to explore other topologies
    if time.time() < deadline - 30:
        bounds = [(-2, 2)] * (n * d)
        try:
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=123, 
                maxiter=50, 
                popsize=10, 
                tol=0.01,
                polish=False
            )
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_x = res_de.x
                
                # Polish the DE result
                if time.time() < deadline - 10:
                    res_polish = minimize(
                        objective, 
                        best_x, 
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12}
                    )
                    if res_polish.fun < best_val and np.isfinite(res_polish.fun):
                        best_val = res_polish.fun
                        best_x = res_polish.x
        except Exception:
            pass

    # If best_x is still None, use the best seed
    if best_x is None:
        best_x = seeds[seed_scores[0][1]].flatten()
    else:
        # Final polish on the very best result
        if time.time() < deadline - 5:
            try:
                res_final = minimize(
                    objective, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res_final.fun < best_val and np.isfinite(res_final.fun):
                    best_val = res_final.fun
                    best_x = res_final.x
            except Exception:
                pass

    # Reshape and center the final result
    final_points = best_x.reshape((n, d))
    final_points = final_points - np.mean(final_points, axis=0)
    
    return final_points
