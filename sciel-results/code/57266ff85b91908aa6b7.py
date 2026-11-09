import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

def construct():
    # 1. Initialization: Structured "Mesh-like" Start
    # Center + 2 rings to encourage interior points and a triangulated mesh
    center = np.array([0.0, 0.0])
    
    # Ring 1: 6 points, radius 1.0
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1 = 1.0
    ring1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    
    # Ring 2: 6 points, radius 2.0, offset by 30 degrees
    angles2 = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    r2 = 2.0
    ring2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    
    # Combine: 1 + 6 + 6 = 13 points
    points = np.vstack([center, ring1, ring2])
    
    # Add small random perturbation to break perfect symmetry and help escape local minima
    rng = np.random.RandomState(123)
    points += rng.normal(0, 0.05, points.shape)
    
    # 2. Compute Initial Delaunay Triangulation (Fixed Topology)
    try:
        delaunay_init = Delaunay(points)
        fixed_triangles = delaunay_init.simplices
    except Exception:
        # Fallback to random if Delaunay fails
        points = rng.uniform(-1, 1, (N, 2))
        delaunay_init = Delaunay(points)
        fixed_triangles = delaunay_init.simplices

    num_tris = len(fixed_triangles)
    
    # 3. Define Objective Function (Soft-Min of Fixed Triangle Areas)
    k_temp = 50.0  # Higher k makes soft-min closer to true min
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 2)
        
        # Compute areas of the fixed triangles
        areas = np.zeros(num_tris)
        # Gradients of areas with respect to points
        # We need dA_i / dp_j for each triangle i and point j in it.
        # A = 0.5 * |cross(p2-p1, p3-p1)|
        # Let's compute signed area gradients first.
        # grad_p1 = 0.5 * perp(p3-p2)
        # grad_p2 = 0.5 * perp(p1-p3)
        # grad_p3 = 0.5 * perp(p2-p1)
        
        grad_areas = np.zeros((num_tris, N, 2))
        
        for i, tri_idx in enumerate(fixed_triangles):
            p1 = pts[tri_idx[0]]
            p2 = pts[tri_idx[1]]
            p3 = pts[tri_idx[2]]
            
            # Signed Area
            cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            area_i = 0.5 * abs(cross)
            areas[i] = area_i
            
            # Gradients of signed area
            v32 = p3 - p2
            v13 = p1 - p3
            v21 = p2 - p1
            
            g1 = 0.5 * np.array([-v32[1], v32[0]])
            g2 = 0.5 * np.array([-v13[1], v13[0]])
            g3 = 0.5 * np.array([-v21[1], v21[0]])
            
            # Sign of the area
            sign = 1.0 if cross >= 0 else -1.0
            
            # Gradient of absolute area = sign * gradient of signed area
            grad_areas[i, tri_idx[0]] = sign * g1
            grad_areas[i, tri_idx[1]] = sign * g2
            grad_areas[i, tri_idx[2]] = sign * g3
            
        # Soft Min Calculation
        # f = -1/k * log(sum(exp(-k * A_i)))
        # We want to maximize f, so we minimize -f = 1/k * log(sum(exp(-k * A_i)))
        
        # Numerical stability: subtract max area? No, we want exp(-k*A).
        # If A is large, exp(-k*A) is small.
        # If A is small, exp(-k*A) is large.
        # To avoid underflow for large A, we can just compute.
        # To avoid overflow for small A (if k*A is large negative? No, -k*A is negative).
        # Wait, if A is very small, -k*A is close to 0, exp is close to 1.
        # If A is large, -k*A is large negative, exp is close to 0.
        # So no overflow risk for positive A and k.
        
        exp_vals = np.exp(-k_temp * areas)
        S = np.sum(exp_vals)
        
        if S == 0:
            return 0.0, np.zeros_like(x_flat)
            
        weights = exp_vals / S
        
        # Gradient of -f (which is 1/k * log(S))
        # d(1/k * log(S)) / dp = 1/k * (1/S) * dS/dp
        # dS/dp = sum( -k * exp(-k*A_i) * dA_i/dp )
        # d(-f)/dp = 1/k * 1/S * sum( -k * exp(-k*A_i) * grad_A_i )
        # d(-f)/dp = - sum( (exp(-k*A_i)/S) * grad_A_i )
        # d(-f)/dp = - sum( weights[i] * grad_A_i )
        
        total_grad = np.sum(weights[:, np.newaxis, np.newaxis] * grad_areas, axis=0)
        
        grad_obj = -total_grad
        
        val = (1.0 / k_temp) * np.log(S)
        
        return val, grad_obj.flatten()

    # 4. Optimization
    x0 = points.flatten()
    
    res = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    final_points = res.x.reshape(N, 2)
    
    # 5. Normalization
    try:
        hull = Delaunay(final_points).convex_hull
        hull_pts = final_points[hull]
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > 0:
            scale = 1.0 / np.sqrt(area)
            final_points = final_points * scale
    except Exception:
        pass
        
    return final_points

# EVOLVE-BLOCK-END
