import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by triangle area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area / AREA

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9

def _params_to_points(params):
    """Convert barycentric coordinates (3N) to Cartesian points."""
    # params is shape (3, N) or (N, 3)? Let's use (N, 3) for clarity: rows are points, cols are barycentric coords
    # Actually, L-BFGS-B prefers a flat vector.
    # Let's define params as shape (N, 3) where each row sums to 1 (implicitly handled by optimization bounds/constraints? No, L-BFGS-B handles bounds).
    # We will optimize over (N, 3) variables, but constrain them to sum to 1?
    # Easier: Optimize over (N, 2) Cartesian coordinates with constraints?
    # The prompt suggests barycentric.
    # Let's use barycentric coordinates (u, v, w) for each point, where u+v+w=1, u,v,w >= 0.
    # We can drop 'w' since w = 1 - u - v.
    # So we optimize 2N variables: u_i, v_i for i=0..N-1.
    # Constraints: u_i >= 0, v_i >= 0, u_i + v_i <= 1.
    # This is a bounded domain! Perfect for L-BFGS-B.
    
    u = params[:N]
    v = params[N:]
    
    points = np.zeros((N, 2))
    for i in range(N):
        w = 1.0 - u[i] - v[i]
        points[i] = V0 * w + V1 * u[i] + V2 * v[i]
    return points

def _soft_min_area(params, beta=100.0):
    """
    Objective function: negative of the soft-minimum triangle area.
    We maximize min area, so we minimize -min area.
    Soft min: -1/beta * log(sum(exp(-beta * a_i)))
    """
    points = _params_to_points(params)
    
    # Compute all triangle areas
    areas = []
    for i in range(N):
        pi = points[i]
        for j in range(i + 1, N):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, N):
                pk = points[k]
                v2 = pk - pi
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft min approximation
    # To avoid overflow, shift by min
    min_a = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = min_a - (1.0 / beta) * log_sum
    
    # Return negative to maximize
    return -soft_min / AREA

def _soft_min_area_grad(params, beta=100.0):
    """
    Numerical gradient for the soft-min objective.
    """
    eps = 1e-5
    grad = np.zeros_like(params)
    f0 = _soft_min_area(params, beta)
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        f_p = _soft_min_area(params_p, beta)
        grad[i] = (f_p - f0) / eps
    return grad

def _generate_random_barycentric(rng):
    """Generate random points in barycentric coordinates (u, v) with u,v >= 0, u+v <= 1."""
    params = np.zeros(2 * N)
    for i in range(N):
        u = rng.random()
        v = rng.random() * (1.0 - u)
        params[i] = u
        params[N + i] = v
    return params

def _generate_structured_barycentric():
    """Generate structured initial configuration in barycentric coords."""
    # Vertices: (0,0)->(0,0), (1,0)->(1,0), (0,1)->(0,1) in (u,v) space?
    # V0 is (0,0) in barycentric (w=1, u=0, v=0) -> (u=0, v=0)
    # V1 is (1,0) in barycentric (w=0, u=1, v=0) -> (u=1, v=0)
    # V2 is (0,1) in barycentric (w=0, u=0, v=1) -> (u=0, v=1)
    
    points = [
        (0.0, 0.0), # V0
        (1.0, 0.0), # V1
        (0.0, 1.0), # V2
        (0.5, 0.0), # Mid V0-V1
        (0.0, 0.5), # Mid V0-V2
        (0.5, 0.5), # Mid V1-V2
        (1/3, 1/3), # Centroid
        (0.25, 0.0),
        (0.75, 0.0),
        (0.25, 0.25),
        (0.5, 0.25)
    ]
    
    params = np.zeros(2 * N)
    for i in range(N):
        params[i] = points[i][0]
        params[N + i] = points[i][1]
    return params

def _local_lbfgsb(params, time_limit, rng):
    """Run L-BFGS-B from a given starting point."""
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    
    # Note: L-BFGS-B does not handle the constraint u+v <= 1 directly with simple bounds.
    # We need to map the domain or use a different approach.
    # Alternative: Optimize Cartesian coordinates with penalty for out-of-bounds?
    # Or use barycentric but enforce u+v<=1 via a change of variables?
    # Let u' = u, v' = v * (1-u). Then u' in [0,1], v' in [0,1].
    # u = u', v = v' * (1-u').
    # This maps the unit square to the simplex.
    
    def transform_params_to_barycentric(p_prime):
        u = p_prime[:N]
        v_prime = p_prime[N:]
        v = v_prime * (1.0 - u)
        return np.concatenate([u, v])
        
    def objective(p_prime):
        bary = transform_params_to_barycentric(p_prime)
        return _soft_min_area(bary, beta=50.0)
        
    # We don't have an easy gradient for the transformed space, so we'll use numerical grad in the transformed space
    # Or just let L-BFGS-B estimate the gradient? It's expensive but robust.
    # Given 11 points, 22 variables. Numerical grad is 22 function evals per step.
    # Function eval is O(N^3) = 11^3 = 1331 ops. Very fast.
    
    start_time = time.time()
    max_iter = 1000
    
    res = minimize(objective, params, method='L-BFGS-B', bounds=[(0.0, 1.0)] * (2*N), 
                   options={'maxiter': max_iter, 'maxfun': 5000, 'disp': False})
    
    # Check time
    if time.time() > time_limit:
        # If we timed out, we might not have the best result, but L-BFGS-B usually finishes fast
        pass
        
    best_params_prime = res.x
    best_bary = transform_params_to_barycentric(best_params_prime)
    best_points = _params_to_points(best_bary)
    best_val = -res.fun
    return best_points, best_val

def _refine_with_coord_descent(points, time_limit, rng):
    """Refine points using coordinate descent on the exact min-area function."""
    current = points.copy()
    current_min = _min_triangle_area(current)
    best_points = current.copy()
    best_min = current_min
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_size in step_sizes:
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                orig = current[i].copy()
                # Try small moves
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    new_p = orig + step_size * np.array([np.cos(angle), np.sin(angle)])
                    if _is_inside_triangle(new_p):
                        candidate = current.copy()
                        candidate[i] = new_p
                        new_min = _min_triangle_area(candidate)
                        if new_min > current_min:
                            current[i] = new_p
                            current_min = new_min
                            improved = True
                            if new_min > best_min:
                                best_min = new_min
                                best_points = current.copy()
                            break
    return best_points, best_min

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Structured start + L-BFGS-B
    if time.time() < deadline - 10:
        params = _generate_structured_barycentric()
        points, val = _local_lbfgsb(params, deadline - 5, rng)
        if val > best_min:
            best_min = val
            best_points = points.copy()
    
    # Strategy 2: Multiple random restarts with L-BFGS-B
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        params = _generate_random_barycentric(rng)
        points, val = _local_lbfgsb(params, deadline - 5, rng)
        if val > best_min:
            best_min = val
            best_points = points.copy()
            
    # Strategy 3: Refine best result with coordinate descent on exact objective
    if best_points is not None and time.time() < deadline - 5:
        refined_points, refined_min = _refine_with_coord_descent(best_points, deadline - 2, rng)
        if refined_min > best_min:
            best_min = refined_min
            best_points = refined_points.copy()
            
    # Fallback
    if best_points is None:
        # Use a simple grid
        points = []
        for i in range(4):
            for j in range(4):
                if i + j < 4:
                    u = i / 3.0
                    v = j / 3.0
                    w = 1 - u - v
                    p = V0 * w + V1 * u + V2 * v
                    points.append(p)
        # Add centroid
        points.append((V0 + V1 + V2) / 3.0)
        best_points = np.array(points[:N])
        
    return best_points
# EVOLVE-BLOCK-END
