"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

S3 = np.sqrt(3.0)
# Equilateral triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
AREA_EQ = S3 / 4.0  # area of the unit side equilateral triangle

# Precompute all pairs of indices
PAIRS = list(combinations(range(N), 2))
N_PAIRS = len(PAIRS)

def compute_areas(points):
    """Compute areas of all triangles formed by 3 points.
    Returns array of shape (N_COMB,)."""
    areas = np.empty((N_COMB,), dtype=np.float64)
    # We need to compute area for each combination of 3 points
    # Using the cross product formula: 0.5 * |cross(p2-p1, p3-p1)|
    # For efficiency, we can vectorize over the third point
    # But let's just do it in a loop for clarity, or use a smarter vectorization
    
    # Actually, let's compute all pairwise vectors first? No, that's 11*11 = 121 vectors.
    # Then area(i,j,k) = 0.5 * |cross(v_ij, v_ik)|
    # Let's precompute pairwise differences
    # points shape (N, 2)
    # v_ij = points[j] - points[i]
    
    # A simpler vectorized approach for N=11:
    # There are 165 combinations.
    # We can iterate over i, then j > i, then k > j.
    # But let's try to vectorize the inner loop.
    
    idx = 0
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            p_j = points[j]
            v_ij = p_j - pi
            for k in range(j + 1, N):
                p_k = points[k]
                v_ik = p_k - pi
                cross = v_ij[0] * v_ik[1] - v_ij[1] * v_ik[0]
                areas[idx] = 0.5 * abs(cross)
                idx += 1
    return areas

N_COMB = 165

def min_area(points):
    """Return the minimum triangle area."""
    areas = compute_areas(points)
    return np.min(areas)

def objective(x):
    """Objective function for optimization: negative of min area."""
    points = x.reshape((N, 2))
    # Check if points are inside the triangle (with a small margin to avoid boundary issues during optimization)
    # If not, return a large penalty
    if not is_inside(points):
        return 1e6
    areas = compute_areas(points)
    return -np.min(areas)

def is_inside(points):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Barycentric coordinates or half-plane checks
    # Triangle vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2)
    # Edges:
    # AB: y >= 0
    # BC: x - 0.5 <= y / sqrt(3)  =>  sqrt(3)*y - x + 0.5 >= 0  =>  x - sqrt(3)*y - 0.5 <= 0
    # CA: x >= 0.5 - y / sqrt(3)  =>  sqrt(3)*y + x - 0.5 >= 0  =>  -x + sqrt(3)*y + 0.5 >= 0
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Margin to keep points strictly inside for numerical stability in optimization
    margin = 1e-8
    cond1 = y >= -margin
    cond2 = (x - S3 * y - 0.5) <= margin
    cond3 = (-x + S3 * y + 0.5) >= -margin
    
    return np.all(cond1) and np.all(cond2) and np.all(cond3)

def constraints(x):
    """Constraints for SLSQP: points must be inside the triangle."""
    points = x.reshape((N, 2))
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    cons = []
    # y >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1], 'jac': None})
    # x - sqrt(3)*y - 0.5 <= 0  =>  0.5 - x + sqrt(3)*y >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 0.5 - x[2*i] + S3 * x[2*i+1], 'jac': None})
    # -x + sqrt(3)*y + 0.5 >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -x[2*i] + S3 * x[2*i+1] + 0.5, 'jac': None})
    
    return cons

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial point: center of triangle
    center = np.array([0.5, S3 / 6.0])
    
    # Generate initial points: one at center, others at vertices and edge midpoints?
    # Or just random inside.
    # Let's use a deterministic initial configuration that is spread out.
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Edge midpoints: (0.5, 0), (0.75, sqrt(3)/4), (0.25, sqrt(3)/4)
    # Center: (0.5, sqrt(3)/6)
    # That's 7 points. We need 11.
    # Let's add 4 more points.
    
    # Better initial guess:
    # Use a grid or known good configuration.
    # Let's try:
    # 1: (0.5, 0.0)
    # 2: (0.5, 0.1)
    # 3: (0.5, 0.2)
    # 4: (0.5, 0.3)
    # 5: (0.5, 0.4)
    # 6: (0.5, 0.5)
    # 7: (0.5, 0.6)
    # 8: (0.5, 0.7)
    # 9: (0.5, 0.8)
    # 10: (0.5, 0.9)
    # 11: (0.5, 1.0)
    # This is bad because they are collinear.
    
    # Let's use a random restart strategy with SLSQP.
    
    best_points = None
    best_val = -1e9
    
    # Precompute constraint functions for speed? SLSQP with lambdas is slow.
    # Let's define a faster constraint function.
    
    def constraints_fast(x):
        points = x.reshape((N, 2))
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Return array of inequality constraints (all must be >= 0)
        c1 = y_coords
        c2 = 0.5 - x_coords + S3 * y_coords
        c3 = -x_coords + S3 * y_coords + 0.5
        
        return np.concatenate([c1, c2, c3])
    
    def jac_constraints(x):
        # Jacobian of constraints
        # c1: y_i
        # c2: 0.5 - x_i + sqrt(3)*y_i
        # c3: -x_i + sqrt(3)*y_i + 0.5
        J = np.zeros((3 * N, 2 * N))
        for i in range(N):
            # c1_i = y_i
            J[3*i, 2*i+1] = 1.0
            # c2_i = 0.5 - x_i + sqrt(3)*y_i
            J[3*i+N, 2*i] = -1.0
            J[3*i+N, 2*i+1] = S3
            # c3_i = -x_i + sqrt(3)*y_i + 0.5
            J[3*i+2*N, 2*i] = -1.0
            J[3*i+2*N, 2*i+1] = S3
        return J
    
    cons = [{'type': 'ineq', 'fun': constraints_fast, 'jac': jac_constraints}]
    
    # Try a few random restarts
    np.random.seed(42)
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random initial points inside the triangle
        points = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v < 1.0:
                    x = u + 0.5 * v
                    y = (S3 / 2.0) * v
                    points[i] = [x, y]
                    break
        
        x0 = points.flatten()
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=cons, 
                bounds=[(0.0, 1.0)] * N + [(0.0, S3/2.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if result.success:
                points_res = result.x.reshape((N, 2))
                if is_inside(points_res):
                    val = -result.fun
                    if val > best_val:
                        best_val = val
                        best_points = points_res
        except Exception as e:
            pass
    
    # If we didn't find anything, use the center
    if best_points is None:
        best_points = np.tile(center, (N, 1))
    
    # Final polish: try to improve the best point found with more iterations
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                constraints=cons, 
                bounds=[(0.0, 1.0)] * N + [(0.0, S3/2.0)] * N,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if result.success:
                points_res = result.x.reshape((N, 2))
                if is_inside(points_res):
                    val = -result.fun
                    if val > best_val:
                        best_val = val
                        best_points = points_res
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
