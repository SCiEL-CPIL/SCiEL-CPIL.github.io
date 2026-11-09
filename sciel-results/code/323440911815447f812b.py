import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize, minimize_scalar
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

    # Better initial configurations
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
    
    def tetrahedron():
        """Return 4 vertices of regular tetrahedron on unit sphere."""
        verts = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]
    
    def generate_initial_configs():
        """Generate multiple initial configurations to try."""
        configs = []
        
        # Config 1: Icosahedron + 2 points at opposite poles
        ico = icosahedron()
        extra1 = np.array([[0, 0, 1.0]])
        extra2 = np.array([[0, 0, -1.0]])
        config1 = np.vstack([ico, extra1, extra2])
        configs.append(config1)
        
        # Config 2: Icosahedron + 2 points at random positions
        np.random.seed(42)
        extra1 = np.random.randn(1, 3)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(1, 3)
        extra2 = extra2 / np.linalg.norm(extra2)
        config2 = np.vstack([ico, extra1, extra2])
        configs.append(config2)
        
        # Config 3: Two tetrahedra (one rotated)
        tet1 = tetrahedron()
        tet2 = tetrahedron()
        # Rotate tet2 by 45 degrees around z-axis
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        tet2 = tet2 @ R.T
        # Scale tet2 slightly to avoid overlap
        tet2 = tet2 * 0.95
        config3 = np.vstack([tet1, tet2, tet1[:2], tet2[:2]])
        configs.append(config3)
        
        # Config 4: Random configuration on sphere
        np.random.seed(123)
        random_pts = np.random.randn(14, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1)[:, np.newaxis]
        configs.append(random_pts)
        
        # Config 5: Icosahedron with 2 points at midpoints of edges
        ico = icosahedron()
        # Find edge midpoints
        mid1 = (ico[0] + ico[1]) / 2
        mid1 = mid1 / np.linalg.norm(mid1)
        mid2 = (ico[2] + ico[3]) / 2
        mid2 = mid2 / np.linalg.norm(mid2)
        config5 = np.vstack([ico, mid1.reshape(1,3), mid2.reshape(1,3)])
        configs.append(config5)
        
        return configs
    
    # Bounds for each coordinate
    bounds = [(-2, 2)] * (n * d)
    
    best_x = None
    best_val = np.inf
    
    # Try each initial configuration
    initial_configs = generate_initial_configs()
    
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        x0 = config.flatten()
        
        # First, do a local refinement with Nelder-Mead (no bounds needed if we center)
        try:
            res_local = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_local.fun < best_val and np.isfinite(res_local.fun):
                best_val = res_local.fun
                best_x = res_local.x
        except Exception:
            pass
        
        # Then try differential evolution from this starting point
        if time.time() > deadline - 20:
            break
            
        try:
            # Use a smaller population and fewer iterations for speed
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=idx + 100, 
                maxiter=80, 
                popsize=12, 
                tol=0.005,
                polish=True,
                init='latinhypercube'
            )
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass
    
    # If we still have time, run a few more DE runs with different seeds
    if time.time() < deadline - 15:
        for s in [1, 2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 5:
                break
                
            try:
                res = differential_evolution(
                    objective, 
                    bounds, 
                    seed=s, 
                    maxiter=100, 
                    popsize=15, 
                    tol=0.003,
                    polish=True
                )
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                break
    
    # Final refinement with Nelder-Mead on the best found
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_refine = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_refine.fun < best_val and np.isfinite(res_refine.fun):
                best_val = res_refine.fun
                best_x = res_refine.x
        except Exception:
            pass
    
    # If best_x is still None, use a fallback
    if best_x is None:
        best_x = initial_configs[0].flatten()
    
    # Reshape and center the final result
    final_points = best_x.reshape((n, d))
    final_points = final_points - np.mean(final_points, axis=0)
    
    # Normalize to unit sphere for better conditioning (optional but can help)
    norms = np.linalg.norm(final_points, axis=1)
    valid_norms = norms[norms > 1e-10]
    if len(valid_norms) > 0:
        # Scale so that the average radius is 1
        avg_norm = np.mean(valid_norms)
        if avg_norm > 1e-10:
            final_points = final_points / avg_norm
    
    return final_points
