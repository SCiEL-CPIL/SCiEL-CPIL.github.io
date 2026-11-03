"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _compute_covering_radius(points):
    """Compute the covering radius (in degrees) for points on unit sphere."""
    try:
        hull = ConvexHull(points)
        max_cos = 1.0
        for simplex in hull.simplices:
            verts = points[simplex]
            if len(verts) == 3:
                v1, v2, v3 = verts
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                dist = np.abs(np.dot(normal, v1))
                dist = np.clip(dist, -1.0, 1.0)
                max_cos = min(max_cos, dist)
            else:
                # Fallback for non-triangular facets (rare in 3D hulls)
                v1, v2, v3 = verts[0], verts[1], verts[2]
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal = normal / norm
                dist = np.abs(np.dot(normal, v1))
                dist = np.clip(dist, -1.0, 1.0)
                max_cos = min(max_cos, dist)
        
        if max_cos > 1.0:
            max_cos = 1.0
        if max_cos < -1.0:
            max_cos = -1.0
        
        radius_rad = np.arccos(max_cos)
        return np.degrees(radius_rad)
    except Exception:
        return 180.0

def _generate_grid(n_lat=50, n_lon=100):
    """Generate a grid of points on the sphere."""
    lats = np.linspace(np.pi/2, -np.pi/2, n_lat)
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    lats, lons = np.meshgrid(lats, lons, indexing='ij')
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)
    grid = np.column_stack([x, y, z])
    return grid

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial points
    points = _fibonacci_spiral(N)
    
    # Grid for surrogate objective
    grid = _generate_grid(40, 80)
    
    # Soft-max parameter
    beta = 20.0
    
    def objective_and_grad(x):
        P = x.reshape(N, 3)
        
        # Compute distances D_ij: (N, M)
        dots = P @ grid.T
        dots = np.clip(dots, -1.0, 1.0)
        D = np.arccos(dots)
        
        # d_j = min_i D_ij
        d = np.min(D, axis=0)
        
        # Soft max of d
        d_max = np.max(d)
        exp_d = np.exp(beta * (d - d_max))
        F = (d_max + np.log(np.sum(exp_d)) / beta)
        
        # Gradient
        sum_exp = np.sum(exp_d)
        weights = exp_d / sum_exp
        
        k_j = np.argmin(D, axis=0)
        
        grad = np.zeros((N, 3))
        
        cos_val = dots[k_j, np.arange(len(d))]
        sin_val = np.sqrt(np.maximum(1e-12, 1 - cos_val**2))
        
        V = -grid / sin_val[:, np.newaxis]
        
        for j in range(len(d)):
            k = k_j[j]
            p_k = P[k]
            v_j = V[j]
            dot_vp = np.dot(v_j, p_k)
            v_tan = v_j - dot_vp * p_k
            
            grad[k] += (weights[j] / beta) * v_tan
            
        return F, grad.flatten()
        
    x0 = points.flatten()
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
        
        if res.success or res.x is not None:
            points_opt = res.x.reshape(N, 3)
            # Renormalize
            norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
            points_opt = points_opt / norms
            
            r_opt = _compute_covering_radius(points_opt)
            
            if r_opt < best_radius:
                best_radius = r_opt
                best_points = points_opt.copy()
                
    except Exception as e:
        pass
        
    return best_points
# EVOLVE-BLOCK-END
