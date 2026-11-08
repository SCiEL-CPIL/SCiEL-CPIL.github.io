import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices
TRI_IDX = np.array(list(combinations(range(N), 3)))
M = len(TRI_IDX)  # C(11,3) = 165

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    if y < 0:
        y = 0
    if y > H:
        y = H
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(seed):
    """Generate random points inside the triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            a = 1 - u - v
            b = u
            c = v
            x = b * 1.0 + c * 0.5
            y = c * H
            points.append([x, y])
    return np.array(points)

def generate_maxmin_initial():
    """Generate initial points using a max-min distance heuristic."""
    # Start with vertices
    points = [V0.copy(), V1.copy(), V2.copy()]
    rng = np.random.RandomState(12345)
    
    for _ in range(N - 3):
        # Generate many candidates and pick the one that maximizes min distance to existing points
        best_candidate = None
        best_min_dist = -1
        
        for _ in range(2000):
            u, v = rng.random(2)
            if u + v < 1:
                a = 1 - u - v
                b = u
                c = v
                x = b * 1.0 + c * 0.5
                y = c * H
                candidate = np.array([x, y])
                
                # Compute min distance to existing points
                dists = np.array([np.linalg.norm(candidate - p) for p in points])
                min_dist = np.min(dists)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
        
        points.append(best_candidate)
    
    return np.array(points)

def objective_and_grad(flat_points):
    """
    Objective: negative of the minimum triangle area (to minimize).
    Also returns gradient for trust-region methods.
    """
    points = flat_points.reshape(N, 2)
    
    # Compute all triangle areas
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Gradient: compute gradient of -min_area with respect to all 2N coordinates
    # For the triangle achieving the minimum, compute its gradient
    i, j, k = TRI_IDX[min_idx]
    
    # Area of triangle (i,j,k) = 0.5 * |cross(pj-pi, pk-pi)|
    # = 0.5 * |(x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)|
    
    # Let's compute gradient analytically
    # f = 0.5 * |a| where a = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # grad f = 0.5 * sign(a) * grad a
    
    sign_a = np.sign(cross[min_idx])
    if sign_a == 0:
        sign_a = 1.0
    
    # a = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # da/dx_i = -(y_k - y_i) - (y_j - y_i) * 0 = -(y_k - y_i)  ... wait, let me be more careful
    # a = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
    # da/dx_i = -y_k + y_i + y_j = y_i + y_j - y_k
    # da/dy_i = -x_j + x_i - x_k + x_i = 2*x_i - x_j - x_k
    # da/dx_j = y_k - y_i
    # da/dy_j = -x_k + x_i = x_i - x_k
    # da/dx_k = -y_j + y_i = y_i - y_j
    # da/dy_k = x_j - x_i
    
    pi_xy = points[i]
    pj_xy = points[j]
    pk_xy = points[k]
    
    grad = np.zeros(2 * N)
    
    # Point i
    grad[2*i] = 0.5 * sign_a * (pi_xy[1] + pj_xy[1] - pk_xy[1])
    grad[2*i+1] = 0.5 * sign_a * (2*pi_xy[0] - pj_xy[0] - pk_xy[0])
    
    # Point j
    grad[2*j] = 0.5 * sign_a * (pk_xy[1] - pi_xy[1])
    grad[2*j+1] = 0.5 * sign_a * (pi_xy[0] - pk_xy[0])
    
    # Point k
    grad[2*k] = 0.5 * sign_a * (pi_xy[1] - pj_xy[1])
    grad[2*k+1] = 0.5 * sign_a * (pj_xy[0] - pi_xy[0])
    
    # We want to maximize min_area, so minimize -min_area
    obj = -min_area
    grad = -grad  # gradient of -min_area
    
    return obj, grad

def objective_only(flat_points):
    """Objective without gradient."""
    obj, _ = objective_and_grad(flat_points)
    return obj

def local_optimize_trust_region(points, deadline, max_iter=100):
    """Local optimization using trust-region reflective method."""
    x0 = points.flatten().copy()
    
    # Bounds: each point must be inside the triangle
    # For simplicity, use loose bounds and project after each step
    bounds = []
    for i in range(N):
        # x range depends on y, but for bounds we use global bounds
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, H))    # y
    
    result = minimize(
        objective_only,
        x0,
        method='trust-ncg',
        bounds=bounds,
        options={'maxiter': max_iter, 'gtol': 1e-8, 'xtol': 1e-10}
    )
    
    opt_points = result.x.reshape(N, 2)
    
    # Project all points to ensure they're inside
    for i in range(N):
        opt_points[i] = project_to_triangle(opt_points[i])
    
    val = min_triangle_area(opt_points)
    return opt_points, val

def local_optimize_hill_climbing(points, max_iter=200, step_size=0.05, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(15):
                if deadline is not None and time.time() > deadline:
                    break
                    
                dx = np.random.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-7:
                break
    
    return points, best_val

def simulated_annealing(points, deadline, initial_temp=0.05, cooling=0.9995, max_iter=3000):
    """Simulated annealing to escape local optima."""
    points = points.copy()
    best_points = points.copy()
    best_val = min_triangle_area(points)
    current_val = best_val
    temp = initial_temp
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        i = np.random.randint(N)
        dx = np.random.uniform(-0.04, 0.04, 2)
        new_p = points[i] + dx
        new_p = project_to_triangle(new_p)
        
        if not is_inside_triangle(new_p, eps=1e-6):
            continue
        
        new_points = points.copy()
        new_points[i] = new_p
        
        new_val = min_triangle_area(new_points)
        delta = new_val - current_val
        
        if delta > 0 or np.random.random() < np.exp(delta / temp):
            points = new_points
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_points = points.copy()
        
        temp *= cooling
    
    return best_points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_value = -1
    
    # Generate multiple initializations
    initializations = [
        generate_structured_initial(),
        generate_maxmin_initial(),
    ]
    for seed in range(6):
        initializations.append(generate_random_points(seed * 137 + 42))
    
    for init_points in initializations:
        if time.time() > deadline:
            break
        
        # First, do trust-region optimization
        if time.time() < deadline - 20:
            tr_points, tr_val = local_optimize_trust_region(init_points, deadline, max_iter=80)
            if tr_val > best_value:
                best_value = tr_val
                best_points = tr_points.copy()
        
        # Then, do hill climbing polish
        if best_points is not None and time.time() < deadline - 10:
            hc_points, hc_val = local_optimize_hill_climbing(best_points, max_iter=150, step_size=0.03, deadline=deadline)
            if hc_val > best_value:
                best_value = hc_val
                best_points = hc_points.copy()
        
        # If we have time, try simulated annealing
        if best_points is not None and time.time() < deadline - 15:
            sa_points, sa_val = simulated_annealing(best_points, deadline, initial_temp=0.03, cooling=0.9997, max_iter=2000)
            if sa_val > best_value:
                best_value = sa_val
                best_points = sa_points.copy()
        
        # Final polish with hill climbing
        if best_points is not None and time.time() < deadline - 5:
            final_points, final_val = local_optimize_hill_climbing(best_points, max_iter=200, step_size=0.01, deadline=deadline)
            if final_val > best_value:
                best_value = final_val
                best_points = final_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
    
    if best_points is None:
        best_points = generate_structured_initial()
    
    return best_points
