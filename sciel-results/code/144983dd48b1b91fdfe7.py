"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to compute covering radius
    def calc_covering_radius(pts):
        """
        Computes the maximum angular distance from any point on the sphere to the nearest site.
        This is equivalent to the covering radius.
        We approximate the sphere with a dense grid to find the worst-case point.
        """
        # Create a dense grid on the sphere for evaluation
        # Using spherical coordinates
        lat = np.linspace(-np.pi/2, np.pi/2, 180)
        lon = np.linspace(0, 2*np.pi, 360)
        lat, lon = np.meshgrid(lat, lon)
        
        # Convert to Cartesian
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Build KDTree of the sites
        tree = cKDTree(pts)
        
        # Query nearest neighbor for each grid point
        # dist is Euclidean distance. 
        # Angular distance theta is related to Euclidean distance d by d = 2*sin(theta/2)
        # So theta = 2*arcsin(d/2)
        dists, _ = tree.query(grid_points)
        
        # Convert Euclidean distances to angular distances
        # Clip to avoid numerical issues with arcsin
        dists_clipped = np.clip(dists / 2.0, 0.0, 1.0)
        angles = 2.0 * np.arcsin(dists_clipped)
        
        return np.max(angles)
    
    # Objective function for optimizer
    def objective(x):
        # x is flattened points
        pts = x.reshape(N, 3)
        
        # Normalize points to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Check for coincident points (distance < 1e-4)
        # If points are too close, penalize heavily to prevent degenerate solutions
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        min_dist = np.min(dists[:, 1])
        if min_dist < 1e-4:
            return 1.0 # Penalize
        
        return calc_covering_radius(pts)
    
    # Constraints: points must be on the unit sphere
    # We handle normalization inside the objective, but SLSQP works better with explicit constraints
    # or we can just let the normalization handle it if the step size is small.
    # However, SLSQP with normalization inside objective is non-smooth at 0.
    # Better approach: Parameterize on sphere? Or use equality constraints.
    # Let's use equality constraints for |p_i|^2 = 1
    
    def sphere_constraints(x):
        pts = x.reshape(N, 3)
        return np.sum(pts**2, axis=1) - 1.0
    
    # Initial guess
    x0 = points.flatten()
    
    # Optimize
    best_x = x0
    best_val = objective(x0)
    
    # Try a few different initializations or just one robust one?
    # The Fibonacci spiral is a good start.
    # We will run SLSQP.
    
    # SLSQP can be slow for N=50 (150 variables).
    # We might need to limit iterations or use a simpler method if it's too slow.
    # Let's try SLSQP with a maxiter limit based on time.
    
    try:
        # Method 'SLSQP' is generally good for constrained problems.
        # 'BFGS' is for unconstrained. Since we normalize, the manifold is smooth.
        # Actually, if we normalize inside objective, the function is not differentiable in the standard sense
        # for the optimizer if it steps off the sphere significantly.
        # However, SLSQP with constraints is the standard way.
        
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            constraints=[{'type': 'eq', 'fun': sphere_constraints}],
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if res.success or res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_points = best_x.reshape(N, 3)
            best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        else:
            best_points = points
            best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
            
    except Exception:
        best_points = points
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        
    # Refinement: Local search around the best found point
    # SLSQP might get stuck. Let's do a few random perturbations and re-optimize?
    # Or just a simple coordinate descent?
    
    # Given time constraints, let's add a simple local polish using random small moves
    # if we have time left.
    if time.time() < deadline - 10:
        current_points = best_points.copy()
        current_score = best_val
        step_size = 0.05
        
        for i in range(100):
            if time.time() > deadline - 5:
                break
                
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                for _ in range(3):
                    # Random direction
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)
                    
                    old_p = current_points[j]
                    new_p = current_points[j] + step_size * rand_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    current_points[j] = new_p
                    new_score = objective(current_points.flatten())
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_points[j] = old_p
                        
            if not improved:
                break
                
        best_points = current_points
        best_val = current_score
        
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
